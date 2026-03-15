"""
Step 5b: Augment Dataset with Handle Features
================================================
Replays the saved MuJoCo states from the demonstration dataset to extract
handle position and door openness — features that the robot needs to know
WHERE the cabinet handle is, but that aren't stored in the parquet files.

Produces an augmented parquet dataset (alongside the original) that the
training script (06) can load automatically.

New features added per timestep (11 dims total):
  handle_pos          (3)  Handle 3D world position
  handle_to_eef_pos   (3)  Handle position relative to end-effector
  door_openness       (1)  Normalized door joint state (0=closed, 1=open)
  handle_xaxis        (3)  Handle body x-axis (door facing direction)
  hinge_direction     (1)  Hinge direction indicator (+1 right, -1 left)

Usage:
    python 05b_augment_handle_data.py

    # Then train with augmented data (auto-detected):
    python 06_train_policy.py --diffusion
"""

import collections
import gzip
import json
import os
import re
import sys

if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np


class _LRUModelCache:
    """Bounded LRU cache for MuJoCo models — prevents unbounded RAM growth."""
    def __init__(self, maxsize=6):
        self._cache = collections.OrderedDict()
        self._maxsize = maxsize

    def __contains__(self, key):
        return key in self._cache

    def __getitem__(self, key):
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


def fix_xml_asset_paths(xml_str):
    """
    Replace absolute mesh/texture paths baked into the model XML (from
    the machine that collected the data) with paths to the local
    robosuite and robocasa installations.
    """
    import robosuite
    import robocasa
    from pathlib import Path

    rs_assets = str(Path(robosuite.__path__[0]) / "models" / "assets")
    rc_assets = str(Path(robocasa.__path__[0]) / "models" / "assets")

    # Common patterns from RoboCasa data collection machines
    xml_str = re.sub(
        r"/opt/conda/envs/robocasa/lib/python3\.\d+/site-packages/robosuite/models/assets",
        rs_assets,
        xml_str,
    )
    xml_str = re.sub(
        r"/root/robocasa/robocasa/models/assets",
        rc_assets,
        xml_str,
    )
    # Catch any remaining absolute robosuite asset paths
    xml_str = re.sub(
        r"/[^\s\"]+/robosuite/models/assets",
        rs_assets,
        xml_str,
    )
    xml_str = re.sub(
        r"/[^\s\"]+/robocasa/models/assets",
        rc_assets,
        xml_str,
    )

    return xml_str


def find_fixture_handle_bodies(model, fixture_name):
    """
    Find MuJoCo body names for the target fixture's door handles.

    HingeCabinet may have one handle (single door) or two (left/right doors).
    Body names follow the pattern: {fixture}_door_handle_main,
    {fixture}_left_door_handle_main, {fixture}_right_door_handle_main.
    """
    handle_bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if fixture_name in name and "handle" in name:
            handle_bodies.append(name)
    return handle_bodies


def find_fixture_door_joints(model, fixture_name):
    """Find door hinge joint names for the target fixture."""
    joints = []
    for i in range(model.njnt):
        jname = model.joint(i).name
        if fixture_name in jname and "door" in jname:
            joints.append((jname, i))
    return joints


def compute_door_openness(model, data, door_joints):
    """
    Compute average normalized door openness (0=closed, 1=fully open).

    Hinge cabinets have two joint conventions:
      - Right-opening: range [0, +1.57], closed at qpos=0 (jmin)
      - Left-opening:  range [-1.57, 0], closed at qpos=0 (jmax)
    We use distance from the CLOSED position (the range bound closest to 0).
    """
    if not door_joints:
        return 0.0
    openness_vals = []
    for jname, jidx in door_joints:
        addr = model.joint(jidx).qposadr[0]
        qpos = data.qpos[addr]
        jrange = model.jnt_range[jidx]
        jmin, jmax = jrange[0], jrange[1]
        if jmax - jmin > 1e-8:
            # Closed position is whichever bound is closer to 0
            if abs(jmin) < abs(jmax):
                # Right-opening: closed=jmin, open=jmax
                norm = abs(qpos - jmin) / (jmax - jmin)
            else:
                # Left-opening: closed=jmax, open=jmin
                norm = abs(qpos - jmax) / (jmax - jmin)
        else:
            norm = 0.0
        openness_vals.append(np.clip(norm, 0.0, 1.0))
    return float(np.mean(openness_vals))


def build_handle_to_joint_map(handle_bodies, door_joints):
    """
    Map each handle body to its associated door joint(s).

    Matching strategy:
    - If only one handle or one set of door joints: all handles share all joints.
    - Otherwise match by 'left'/'right' substring in name.

    Returns dict: {handle_body_name: [(jname, jidx), ...]}
    """
    if len(handle_bodies) == 1 or len(door_joints) == 1:
        return {hb: door_joints for hb in handle_bodies}

    result = {}
    for hb in handle_bodies:
        hb_lower = hb.lower()
        if "left" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "left" in jn.lower()]
        elif "right" in hb_lower:
            matched = [(jn, ji) for jn, ji in door_joints if "right" in jn.lower()]
        else:
            matched = []
        result[hb] = matched if matched else door_joints
    return result


def get_hinge_direction(handle_body, handle_to_joint_map, model):
    """Return hinge direction (+1 right-opening, -1 left-opening) for a handle's door."""
    joints = handle_to_joint_map.get(handle_body, [])
    if not joints:
        return 0.0
    _, jidx = joints[0]
    jrange = model.jnt_range[jidx]
    jmin, jmax = jrange[0], jrange[1]
    return 1.0 if abs(jmin) < abs(jmax) else -1.0


def process_episode(ep_dir, model_cache=None):
    """
    Replay one episode's MuJoCo states and extract handle features.

    Returns:
        dict with keys: handle_pos (N,3), handle_to_eef_pos (N,3), door_openness (N,1)
        or None if the episode can't be processed.
    """
    import mujoco

    ep_meta_path = ep_dir / "ep_meta.json"
    states_path = ep_dir / "states.npz"
    model_path = ep_dir / "model.xml.gz"

    if model_cache is None:
        model_cache = _LRUModelCache()

    if not all(p.exists() for p in [ep_meta_path, states_path, model_path]):
        return None

    # Load episode metadata
    with open(ep_meta_path) as f:
        meta = json.load(f)

    fixture_refs = meta.get("fixture_refs", {})
    fixture_name = fixture_refs.get("fxtr")
    if not fixture_name:
        print(f"    WARNING: No fixture ref in {ep_dir.name}, skipping")
        return None

    # Load model XML
    with gzip.open(model_path, "rb") as f:
        xml_str = f.read().decode("utf-8")
    xml_str = fix_xml_asset_paths(xml_str)

    # Cache models by XML hash to avoid reloading identical models
    xml_hash = hash(xml_str)
    if xml_hash in model_cache:
        model = model_cache[xml_hash]
    else:
        try:
            model = mujoco.MjModel.from_xml_string(xml_str)
        except Exception as e:
            print(f"    WARNING: Failed to load model for {ep_dir.name}: {e}")
            return None
        model_cache[xml_hash] = model

    data = mujoco.MjData(model)

    # Load states via memory-map — avoids copying the full array into RAM
    states_arr = np.load(states_path, mmap_mode="r")["states"]
    num_steps = states_arr.shape[0]

    # Find handle bodies and door joints
    handle_bodies = find_fixture_handle_bodies(model, fixture_name)
    door_joints = find_fixture_door_joints(model, fixture_name)

    if not handle_bodies:
        print(f"    WARNING: No handle bodies found for {fixture_name} in {ep_dir.name}")
        return None

    # Pre-allocate output arrays
    handle_pos_arr = np.zeros((num_steps, 3), dtype=np.float32)
    handle_to_eef_arr = np.zeros((num_steps, 3), dtype=np.float32)
    door_openness_arr = np.zeros((num_steps, 1), dtype=np.float32)
    handle_xaxis_arr = np.zeros((num_steps, 3), dtype=np.float32)
    hinge_dir_arr = np.zeros((num_steps, 1), dtype=np.float32)

    # Map each handle to the door joint(s) it belongs to
    handle_to_joint_map = build_handle_to_joint_map(handle_bodies, door_joints)

    # Switch to the next handle once the current door reaches this openness
    OPEN_THRESHOLD = 0.90

    for t in range(num_steps):
        # Set MuJoCo state: states = [time(1), qpos(nq), qvel(nv), ...]
        data.qpos[:] = states_arr[t, 1:1 + model.nq]
        data.qvel[:] = states_arr[t, 1 + model.nq:1 + model.nq + model.nv]
        mujoco.mj_forward(model, data)

        eef_pos = data.body("gripper0_right_eef").xpos.copy()

        # Per-handle door openness
        per_door = {
            hb: compute_door_openness(model, data, handle_to_joint_map[hb])
            for hb in handle_bodies
        }

        # Active handles: those whose door is NOT yet fully open.
        # If all doors are open, fall back to all handles (episode near end).
        active = [hb for hb in handle_bodies if per_door[hb] < OPEN_THRESHOLD]
        candidates = active if active else handle_bodies

        # Pick the nearest candidate handle
        dists = [np.linalg.norm(data.body(hb).xpos - eef_pos) for hb in candidates]
        target_handle = candidates[int(np.argmin(dists))]

        handle_pos = data.body(target_handle).xpos.copy()
        handle_to_eef = handle_pos - eef_pos
        openness = per_door[target_handle]

        # Handle body x-axis (first column of xmat) — encodes door facing direction
        xmat = data.body(target_handle).xmat.reshape(3, 3)
        handle_xaxis = xmat[:, 0].copy()

        hinge_dir = get_hinge_direction(target_handle, handle_to_joint_map, model)

        handle_pos_arr[t] = handle_pos.astype(np.float32)
        handle_to_eef_arr[t] = handle_to_eef.astype(np.float32)
        door_openness_arr[t, 0] = openness
        handle_xaxis_arr[t] = handle_xaxis.astype(np.float32)
        hinge_dir_arr[t, 0] = hinge_dir

    return {
        "handle_pos": handle_pos_arr,
        "handle_to_eef_pos": handle_to_eef_arr,
        "door_openness": door_openness_arr,
        "handle_xaxis": handle_xaxis_arr,
        "hinge_direction": hinge_dir_arr,
    }


def main():
    from pathlib import Path
    import pyarrow.parquet as pq
    import pyarrow as pa

    print("=" * 60)
    print("  OpenCabinet - Augment Dataset with Handle Features")
    print("=" * 60)

    dataset_path = Path(get_dataset_path())
    print(f"\nDataset: {dataset_path}")

    # Locate parquet data and extras
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        data_dir = dataset_path / "lerobot" / "data"
    extras_dir = dataset_path / "extras"

    chunk_dir = data_dir / "chunk-000"
    if not chunk_dir.exists():
        print(f"ERROR: Chunk directory not found: {chunk_dir}")
        sys.exit(1)

    parquet_files = sorted(chunk_dir.glob("*.parquet"))
    episode_dirs = sorted(extras_dir.glob("episode_*"))

    print(f"Parquet files: {len(parquet_files)}")
    print(f"Episode extras: {len(episode_dirs)}")

    if not episode_dirs:
        print("ERROR: No episode extras found. Dataset may be incomplete.")
        sys.exit(1)

    # Output directory for augmented data
    aug_dir = dataset_path / "augmented"
    aug_dir.mkdir(exist_ok=True)
    print(f"Output: {aug_dir}")

    print_section("Processing Episodes")

    model_cache = _LRUModelCache(maxsize=6)
    total_steps = 0
    processed_eps = 0
    skipped_eps = 0

    # Process each parquet file (each contains one episode)
    for pf_idx, pf in enumerate(parquet_files):
        # Episode index from filename: episode_000000.parquet
        ep_num = int(pf.stem.split("_")[-1])
        ep_dir = extras_dir / f"episode_{ep_num:06d}"

        if not ep_dir.exists():
            print(f"  Episode {ep_num:4d}: no extras dir, skipping")
            skipped_eps += 1
            continue

        # Load original parquet
        table = pq.read_table(pf)
        df = table.to_pandas()
        num_rows = len(df)

        # Process episode
        features = process_episode(ep_dir, model_cache)

        if features is None:
            print(f"  Episode {ep_num:4d}: processing failed, skipping")
            skipped_eps += 1
            continue

        # The parquet may have fewer rows than states (states include initial state)
        # Align: parquet rows correspond to states[1:] (after the initial reset state)
        feat_steps = features["handle_pos"].shape[0]

        if feat_steps - 1 == num_rows:
            # States has one extra (initial state), parquet starts from step 1
            for key in features:
                features[key] = features[key][1:]
        elif feat_steps == num_rows:
            pass  # Already aligned
        elif feat_steps > num_rows:
            # Trim to match parquet
            for key in features:
                features[key] = features[key][:num_rows]
        else:
            # Fewer states than parquet rows — pad with last value
            for key in features:
                pad_count = num_rows - feat_steps
                last_val = features[key][-1:]
                features[key] = np.concatenate(
                    [features[key], np.repeat(last_val, pad_count, axis=0)]
                )

        # Add new columns as list-of-arrays (same format as observation.state)
        df["observation.handle_pos"] = [
            features["handle_pos"][i] for i in range(num_rows)
        ]
        df["observation.handle_to_eef_pos"] = [
            features["handle_to_eef_pos"][i] for i in range(num_rows)
        ]
        df["observation.door_openness"] = [
            features["door_openness"][i] for i in range(num_rows)
        ]
        df["observation.handle_xaxis"] = [
            features["handle_xaxis"][i] for i in range(num_rows)
        ]
        df["observation.hinge_direction"] = [
            features["hinge_direction"][i] for i in range(num_rows)
        ]

        # Save augmented parquet
        out_path = aug_dir / pf.name
        new_table = pa.Table.from_pandas(df)
        pq.write_table(new_table, out_path)
        del df, new_table

        processed_eps += 1
        total_steps += num_rows

        if (pf_idx + 1) % 10 == 0 or pf_idx == 0:
            print(
                f"  Episode {ep_num:4d}: {num_rows:4d} steps, "
                f"handle_pos sample={features['handle_pos'][0]}"
            )

    print_section("Summary")
    print(f"  Processed: {processed_eps} episodes ({total_steps} total steps)")
    print(f"  Skipped:   {skipped_eps} episodes")
    print(f"  Output:    {aug_dir}")
    print(f"  New features per step:")
    print(f"    observation.handle_pos          (3)  Handle world position")
    print(f"    observation.handle_to_eef_pos   (3)  Handle relative to EEF")
    print(f"    observation.door_openness       (1)  Door openness [0-1]")
    print(f"\n  The training script will auto-detect augmented data.")
    print(f"  Run:  python 06_train_policy.py --diffusion")


if __name__ == "__main__":
    main()
