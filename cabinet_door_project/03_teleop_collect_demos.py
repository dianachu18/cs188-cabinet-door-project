"""
Step 3: Teleoperate the Robot to Collect Demonstrations
========================================================
Opens an interactive window where you control the PandaOmron robot
with your keyboard (or SpaceMouse) to open cabinet doors.

This gives you hands-on intuition for the task. Note: in normal mode
this script does NOT save demonstration data to disk. To get training
data, run 04_download_dataset.py to download the pre-collected dataset.

DAgger mode (--dagger):
    The trained policy drives the robot autonomously. Press movement
    keys to override the policy at any time. All (state, action) pairs
    are saved as parquet files compatible with the training pipeline in
    06_train_policy.py. This implements Dataset Aggregation (DAgger) —
    a simple way to improve a policy by collecting corrections.

Usage:
    # Mac users MUST use mjpython for the rendering window
    mjpython 03_teleop_collect_demos.py

    # Linux users
    python 03_teleop_collect_demos.py

    # Use spacemouse instead of keyboard
    python 03_teleop_collect_demos.py --device spacemouse

    # DAgger mode: policy drives, human overrides with keyboard
    mjpython 03_teleop_collect_demos.py --dagger --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # DAgger with custom output directory
    mjpython 03_teleop_collect_demos.py --dagger --checkpoint best_policy.pt --save_dir data/dagger_round2/chunk-000

    Recording:
        Q       - Discard the current episode
"""

import os
import sys

# ── WSLg / XWayland GL setup — re-exec approach ─────────────────────────────
# On WSLg (Windows 11) the .bashrc often sets DISPLAY to a stale VcXsrv-style
# "IP:0" address, and Mesa's D3D12 GPU path fails with gladLoadGL on XWayland.
#
# Setting os.environ inside a running Python process is NOT reliable: C
# libraries (Mesa, GLFW) may read the env at dlopen() time which happens
# during import, before our code runs.  The only guaranteed fix is to restart
# the process with the correct vars already in the OS-level environment.
#
# We use os.execve() to atomically replace this process with an identical one
# that has the correct vars from the very start.  A sentinel env var prevents
# the new process from re-execing again.
if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
    _env = dict(os.environ)
    _changed = False

    if _env.get("WAYLAND_DISPLAY"):
        # WSLg: force XWayland socket display and Mesa software renderer.
        if not _env.get("DISPLAY", "").startswith(":"):
            _env["DISPLAY"] = ":0"
            _changed = True
        if _env.get("GALLIUM_DRIVER") != "llvmpipe":
            _env["GALLIUM_DRIVER"] = "llvmpipe"
            _changed = True
        if _env.get("MESA_GL_VERSION_OVERRIDE") != "4.5":
            _env["MESA_GL_VERSION_OVERRIDE"] = "4.5"
            _changed = True

    if _changed:
        _env["__TELEOP_DISPLAY_OK"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, _env)
    else:
        os.environ["__TELEOP_DISPLAY_OK"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import argparse
import collections
import importlib
import time
from copy import deepcopy

import numpy as np
import robocasa  # noqa: F401 - registers environments including OpenCabinet
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper


# Import policy loading & state extraction from 07_evaluate_policy 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.join(SCRIPT_DIR, "diffusion_policy")
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)

_eval_mod_path = os.path.join(SCRIPT_DIR, "07_evaluate_policy.py")
_spec = importlib.util.spec_from_file_location("eval_policy", _eval_mod_path)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)

load_policy = _eval_mod.load_policy
LiveHandleAugmenter = _eval_mod.LiveHandleAugmenter
extract_state = _eval_mod.extract_state


def save_trajectory_parquet(trajectory, save_dir, episode_index):
    """
    Save a list of {state, action, aug_features} dicts as a parquet file.

    The output schema matches what build_diffusion_dataset in 06_train_policy.py
    expects: columns ``observation.state``, ``action``, and augmented observation
    columns (handle_pos, handle_to_eef_pos, door_openness, handle_xaxis,
    hinge_direction).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    os.makedirs(save_dir, exist_ok=True)

    # Extract raw state (16D) and action (12D)
    raw_states = [step["raw_state"].tolist() for step in trajectory]
    actions = [step["action"].tolist() for step in trajectory]

    columns = {
        "observation.state": raw_states,
        "action": actions,
    }

    # Add augmented features as separate columns (matching 05b output format)
    if trajectory[0].get("aug_features") is not None:
        handle_pos = [step["aug_features"][:3].tolist() for step in trajectory]
        handle_to_eef = [step["aug_features"][3:6].tolist() for step in trajectory]
        door_openness = [step["aug_features"][6:7].tolist() for step in trajectory]
        handle_xaxis = [step["aug_features"][7:10].tolist() for step in trajectory]
        hinge_dir = [step["aug_features"][10:11].tolist() for step in trajectory]

        columns["observation.handle_pos"] = handle_pos
        columns["observation.handle_to_eef_pos"] = handle_to_eef
        columns["observation.door_openness"] = door_openness
        columns["observation.handle_xaxis"] = handle_xaxis
        columns["observation.hinge_direction"] = hinge_dir

    table = pa.table(columns)

    path = os.path.join(save_dir, f"episode_{episode_index:06d}.parquet")
    pq.write_table(table, path)
    return path


def _env_action_to_lerobot(env_action):
    """Reorder action from env/HDF5 format to LeRobot/parquet format.

    Env format:     [eef_pos(3), eef_rot(3), gripper(1), base(4), mode(1)]
    LeRobot format: [base(4), mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    """
    lerobot_action = np.zeros(12, dtype=np.float32)
    lerobot_action[0:4] = env_action[7:11]    # base_motion
    lerobot_action[4:5] = env_action[11:12]   # control_mode
    lerobot_action[5:8] = env_action[0:3]     # eef_position
    lerobot_action[8:11] = env_action[3:6]    # eef_rotation
    lerobot_action[11:12] = env_action[6:7]   # gripper_close
    return lerobot_action


def _lerobot_action_to_env(action):
    """Reorder action from LeRobot/parquet format to env/HDF5 format.

    LeRobot format: [base(4), mode(1), eef_pos(3), eef_rot(3), gripper(1)]
    Env format:     [eef_pos(3), eef_rot(3), gripper(1), base(4), mode(1)]
    """
    env_action = np.zeros(12, dtype=np.float32)
    env_action[0:3] = action[5:8]     # eef_position
    env_action[3:6] = action[8:11]    # eef_rotation
    env_action[6:7] = action[11:12]   # gripper_close
    env_action[7:11] = action[0:4]    # base_motion
    env_action[11:12] = action[4:5]   # control_mode
    return env_action


def collect_dagger_trajectory(
    env, device, policy_info, torch_device,
    mirror_actions=True, max_fr=30,
):
    """
    Collect a single DAgger trajectory.

    The trained policy drives the robot. When the human presses movement
    keys, their input overrides the policy. All (state, action) pairs are
    recorded regardless of who was in control.

    Supports both simple MLP and diffusion policy (with action chunking).

    Returns:
        (success, trajectory): success bool and list of {raw_state, action,
        aug_features} dicts. Actions are in LeRobot format for training
        compatibility.
    """
    import torch

    policy_type = policy_info["type"]
    state_dim = policy_info["state_dim"]
    action_dim = policy_info["action_dim"]
    model = policy_info["model"]
    n_obs_steps = policy_info.get("n_obs_steps", 1)
    n_action_steps = policy_info.get("n_action_steps", 1)

    obs = env.reset()

    # Initialize handle augmenter for this episode
    augmenter = LiveHandleAugmenter(env)

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    if lang is not None:
        print(f"  Task: {lang}")

    task_completion_hold_count = -1
    device_input = device
    device_input.start_control()

    # Track gripper state
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Dummy step to initialize
    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)

    discard_traj = False
    trajectory = []
    action_buffer = collections.deque()
    obs_history = collections.deque(maxlen=n_obs_steps)
    step_count = 0
    # Grace period: after human input, stay in human mode for this many steps
    # so the policy doesn't immediately reclaim control (especially important
    # for diffusion policy whose predict_action() call blocks briefly).
    HUMAN_GRACE_STEPS = 30  # ~1 second at 30 fps
    human_grace_remaining = 0

    while True:
        start = time.time()

        active_robot = env.robots[device_input.active_robot]

        # Get human input.  Use goal_update_mode="achieved" so the keyboard
        # controller always references the robot's current position, not a
        # stale target from before the policy was driving.
        input_ac_dict = device_input.input2action(
            mirror_actions=mirror_actions, goal_update_mode="achieved"
        )

        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions based on controller type
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]

        # Detect human activity: check if right_delta or base actions are non-zero
        human_input_now = False
        right_delta = input_ac_dict.get("right_delta", None)
        if right_delta is not None and np.any(right_delta != 0):
            human_input_now = True
        base_action = input_ac_dict.get("base", None)
        if base_action is not None and np.any(base_action != 0):
            human_input_now = True

        if human_input_now:
            human_grace_remaining = HUMAN_GRACE_STEPS
        elif human_grace_remaining > 0:
            human_grace_remaining -= 1

        human_active = human_input_now or human_grace_remaining > 0

        # Compute state (with handle augmentation) for both policy and recording
        aug_feats = augmenter.compute(env) if augmenter.handle_bodies else None
        state = extract_state(obs, state_dim, augmented_features=aug_feats)
        obs_history.append(state)

        if human_active:
            # Human override: build action from human input, clear action buffer
            action_buffer.clear()
            env_action = [
                robot.create_action_vector(all_prev_gripper_actions[i])
                for i, robot in enumerate(env.robots)
            ]
            env_action[device_input.active_robot] = (
                active_robot.create_action_vector(action_dict)
            )
            env_action = np.concatenate(env_action)
        else:
            # Policy drives
            if len(action_buffer) == 0:
                if policy_type == "simple_mlp":
                    with torch.no_grad():
                        state_t = torch.from_numpy(state).unsqueeze(0).to(torch_device)
                        lerobot_action = model(state_t).cpu().numpy().squeeze(0)
                    action_buffer.append(lerobot_action)
                else:
                    # Diffusion policy: build obs sequence and run inference
                    while len(obs_history) < n_obs_steps:
                        obs_history.appendleft(obs_history[0])

                    obs_seq = np.stack(list(obs_history), axis=0)  # (n_obs, obs_dim)
                    obs_tensor = (
                        torch.from_numpy(obs_seq)
                        .float()
                        .unsqueeze(0)
                        .to(torch_device)
                    )  # (1, n_obs, obs_dim)

                    with torch.no_grad():
                        result = model.predict_action({"obs": obs_tensor})
                        action_chunk = result["action"].cpu().numpy().squeeze(0)
                        # action_chunk: (n_action_steps, action_dim)

                    for a in action_chunk:
                        action_buffer.append(a)

            lerobot_action = action_buffer.popleft()

            # Convert from LeRobot action order to env action order
            env_action = _lerobot_action_to_env(lerobot_action)

            # Pad/trim to environment action dimension
            env_dim = env.action_dim
            if len(env_action) < env_dim:
                env_action = np.pad(env_action, (0, env_dim - len(env_action)))
            elif len(env_action) > env_dim:
                env_action = env_action[:env_dim]

        # Step the environment
        obs, _, _, _ = env.step(env_action)

        # Record (raw_state, action in LeRobot format, aug_features)
        # Raw state is the 16D proprioception without augmented features
        RAW_STATE_KEYS = [
            "robot0_base_pos", "robot0_base_quat",
            "robot0_base_to_eef_pos", "robot0_base_to_eef_quat",
            "robot0_gripper_qpos",
        ]
        raw_parts = []
        for key in RAW_STATE_KEYS:
            if key in obs and isinstance(obs[key], np.ndarray):
                raw_parts.append(obs[key].flatten())
        raw_state = np.concatenate(raw_parts).astype(np.float32) if raw_parts else np.zeros(16, dtype=np.float32)

        # Convert env action to LeRobot format for saving
        recorded_action = _env_action_to_lerobot(env_action[:12])

        trajectory.append({
            "raw_state": raw_state,
            "action": recorded_action,
            "aug_features": aug_feats,
        })

        # Status line every 10 steps
        step_count += 1
        if step_count % 10 == 0:
            if human_input_now:
                who = "[HUMAN]"
            elif human_grace_remaining > 0:
                who = f"[HUMAN grace={human_grace_remaining}]"
            else:
                who = "[policy]"
            print(f"\r  step {step_count:4d}  {who}  "
                  f"traj_len={len(trajectory)}    ", end="", flush=True)

        # Check for task completion (15 consecutive success steps)
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 14
        else:
            task_completion_hold_count = -1

        # Frame rate limiting
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    # Clear the \r status line
    print()

    success = not discard_traj
    return success, trajectory


def collect_trajectory(env, device, mirror_actions=True, max_fr=30):
    """
    Collect a single teleoperation trajectory.

    This is a simplified version of RoboCasa's collect_human_trajectory
    that avoids the circular import in robocasa.scripts.collect_demos.

    Returns:
        success (bool): Whether the cabinet was opened during the episode.
    """
    env.reset()

    ep_meta = env.get_ep_meta()
    lang = ep_meta.get("lang", None)
    if lang is not None:
        print(f"  Task: {lang}")

    # Counter: task must be successful for 15 consecutive timesteps.
    # Counts down from 14 to 0, then breaks on the next check (= 15 steps total).
    task_completion_hold_count = -1
    device.start_control()
    nonzero_ac_seen = False

    # Track gripper state
    all_prev_gripper_actions = [
        {
            f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
            for robot_arm in robot.arms
            if robot.gripper[robot_arm].dof > 0
        }
        for robot in env.robots
    ]

    # Do a dummy step to initialize
    zero_action = np.zeros(env.action_dim)
    env.step(zero_action)

    discard_traj = False

    while True:
        start = time.time()

        active_robot = env.robots[device.active_robot]

        # Get action from input device
        input_ac_dict = device.input2action(mirror_actions=mirror_actions)

        # None means the user pressed Q (reset signal)
        if input_ac_dict is None:
            discard_traj = True
            break

        action_dict = deepcopy(input_ac_dict)

        # Set arm actions based on controller type
        for arm in active_robot.arms:
            controller_input_type = active_robot.part_controllers[arm].input_type
            if controller_input_type == "delta":
                action_dict[arm] = input_ac_dict[f"{arm}_delta"]
            elif controller_input_type == "absolute":
                action_dict[arm] = input_ac_dict[f"{arm}_abs"]

        # Skip if no meaningful input yet (spacemouse idle)
        if not nonzero_ac_seen:
            is_empty = np.all(action_dict.get("right_delta", np.array([1])) == 0)
            if is_empty:
                continue
            nonzero_ac_seen = True

        # Build full action vector
        env_action = [
            robot.create_action_vector(all_prev_gripper_actions[i])
            for i, robot in enumerate(env.robots)
        ]
        env_action[device.active_robot] = active_robot.create_action_vector(
            action_dict
        )
        env_action = np.concatenate(env_action)

        # Step the environment
        obs, _, _, _ = env.step(env_action)

        # Check for task completion (15 consecutive success steps)
        if task_completion_hold_count == 0:
            break

        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1
            else:
                task_completion_hold_count = 14
        else:
            task_completion_hold_count = -1

        # Frame rate limiting
        if max_fr is not None:
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

    success = not discard_traj
    return success


def _check_display():
    """Exit early with helpful instructions if no X display is available."""
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")

    if wayland:
        # WSLg is running. It provides XWayland at :0, which both pynput
        # (X11-based) and MuJoCo's GLFW viewer need.
        if not display or not display.startswith(":"):
            os.environ["DISPLAY"] = ":0"
        # Force Mesa software rendering (llvmpipe) for the GLFW window.
        # WSLg's Mesa D3D12 GPU path often fails with gladLoadGL; llvmpipe
        # is slower but reliably provides OpenGL 4.5 for the viewer.
        os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")
        os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.5")
        return

    if display:
        # Some X display is claimed — let MuJoCo's own error handling deal
        # with actual render failures.
        return

    # Nothing set at all.
    print("ERROR: This script requires a display (X server) for the MuJoCo viewer")
    print("       and keyboard input. No display environment variable is set.")
    print()
    print("Windows 11 (WSLg — recommended, no extra software needed):")
    print("  WSLg provides a built-in display. If DISPLAY is not set, try:")
    print("  export DISPLAY=:0")
    print()
    print("Windows 10 / VcXsrv (Maybe MacOS too?):")
    print("  1. Launch XLaunch, on 'Extra settings' uncheck 'Native opengl'")
    print("     and check 'Disable access control'")
    print("  2. export DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0")
    print("  3. export LIBGL_ALWAYS_INDIRECT=0")
    print()
    print("Note: steps 01, 02, 04-07 do NOT require a display.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Teleoperate robot for OpenCabinet")
    parser.add_argument(
        "--layout", type=int, default=None, help="Kitchen layout ID (1-60)"
    )
    parser.add_argument(
        "--style", type=int, default=None, help="Kitchen style ID (1-60)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse"],
        help="Input device",
    )
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="Enable DAgger mode: policy drives, human overrides with keyboard",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (.pt) — required with --dagger",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/dagger/chunk-000",
        help="Where to save DAgger trajectories (default: data/dagger/chunk-000)",
    )
    args = parser.parse_args()

    if args.dagger and not args.checkpoint:
        parser.error("--checkpoint is required when using --dagger")

    _check_display()

    print("=" * 60)
    print("  OpenCabinet - Teleoperation Demo Collection")
    print("=" * 60)
    print()

    # Create the environment
    config = {
        "env_name": "OpenCabinet",
        "robots": "PandaOmron",
        "controller_configs": load_composite_controller_config(robot="PandaOmron"),
        "layout_ids": args.layout,
        "style_ids": args.style,
        "translucent_robot": True,
    }

    print("Initializing environment (this may take a moment)...")
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )

    env = VisualizationWrapper(env)

    # Initialize input device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(env=env, pos_sensitivity=4.0, rot_sensitivity=4.0)
    elif args.device == "spacemouse":
        import robocasa.macros as macros
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            env=env,
            pos_sensitivity=4.0,
            rot_sensitivity=4.0,
            vendor_id=macros.SPACEMOUSE_VENDOR_ID,
            product_id=macros.SPACEMOUSE_PRODUCT_ID,
        )

    # ── DAgger mode setup ──────────────────────────────────────────────────
    if args.dagger:
        import torch

        if not os.path.exists(args.checkpoint):
            print(f"ERROR: Checkpoint not found: {args.checkpoint}")
            print("Train a policy first with:  python 06_train_policy.py")
            sys.exit(1)

        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_info = load_policy(args.checkpoint, torch_device)

        print(f"DAgger mode enabled")
        print(f"  Checkpoint:  {args.checkpoint}")
        print(f"  Policy type: {policy_info['type']}")
        print(f"  State dim: {policy_info['state_dim']},  Action dim: {policy_info['action_dim']}")
        if policy_info["type"] == "diffusion_unet":
            print(f"  Obs steps: {policy_info['n_obs_steps']}, Action steps: {policy_info['n_action_steps']}")
        print(f"  Save dir:  {args.save_dir}")
        print()
        print("The policy will drive the robot automatically.")
        print("Press movement keys to override. Press Q to discard an episode.")
        print()

    # ── Episode loop ─────────────────────────────────────────────────────
    if not args.dagger:
        print("\nReady! Move the robot to open the cabinet door.")
        print("Press Q when done with each episode.\n")

    episode = 0
    saved_count = 0
    try:
        while True:
            episode += 1
            print(f"--- Episode {episode} ---")

            if args.dagger:
                success, trajectory = collect_dagger_trajectory(
                    env, device, policy_info, torch_device,
                    mirror_actions=True, max_fr=30,
                )
                if success and trajectory:
                    path = save_trajectory_parquet(
                        trajectory, args.save_dir, saved_count
                    )
                    saved_count += 1
                    print(f"  Result: saved {len(trajectory)} steps -> {path}")
                else:
                    print(f"  Result: Discarded")
            else:
                success = collect_trajectory(
                    env, device, mirror_actions=True, max_fr=30
                )
                status = "SUCCESS" if success else "Discarded"
                print(f"  Result: {status}")

            print()
    except KeyboardInterrupt:
        if args.dagger:
            print(f"\nDAgger collection ended. Saved {saved_count} episodes "
                  f"to {args.save_dir}")
        else:
            print("\nTeleoperation ended.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
