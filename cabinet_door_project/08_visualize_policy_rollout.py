"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

Supports both simple MLP and diffusion U-Net policies (auto-detected
from checkpoint).

This is your primary debugging tool: watch exactly where and why the policy
fails — does it reach for the handle? Does it grasp? Does it pull correctly?

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to a video file, works without a display

Usage:
    # Watch live in a window — MLP policy
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Watch live — diffusion policy (auto-detected)
    python 08_visualize_policy_rollout.py --checkpoint /tmp/cabinet_diffusion_checkpoints/best_diffusion_policy.pt

    # Save to video only (no display needed — works headless / in notebooks)
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen

    # Run 3 episodes, slow down playback so you can follow along
    python 08_visualize_policy_rollout.py --checkpoint ... --num_episodes 3 --max_steps 200

    # Mac users must use mjpython for the on-screen window
    mjpython 08_visualize_policy_rollout.py --checkpoint ...
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────────
# We peek at sys.argv *before* argparse so we can configure the GL backend
# before any library is imported.  Wrong GL backend = gladLoadGL error.
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    # Off-screen mode: use Mesa's software osmesa renderer.
    # EGL is the default on headless Linux but fails on WSL2 (no /dev/dri).
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
    # On-screen mode: re-exec with correct display vars baked into the OS
    # environment so Mesa (GLFW) sees them before any C library initializes.
    # On WSLg the .bashrc often sets a stale VcXsrv-style DISPLAY that
    # breaks GLFW; os.execve() restarts the process cleanly.
    if sys.platform == "linux" and "__TELEOP_DISPLAY_OK" not in os.environ:
        _env = dict(os.environ)
        _changed = False
        if _env.get("WAYLAND_DISPLAY"):
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
import time

import numpy as np
import robocasa  # noqa: F401 — registers OpenCabinet environment
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# Add diffusion_policy to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.join(SCRIPT_DIR, "diffusion_policy")
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)


# ── Policy loading (supports both MLP and diffusion) ────────────────────────

def load_policy(checkpoint_path, device):
    """Load a policy checkpoint (MLP or Diffusion U-Net, auto-detected)."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_type = ckpt.get("policy_type", "simple_mlp")

    if policy_type == "diffusion_unet":
        return _load_diffusion_policy(ckpt, device)
    else:
        return _load_simple_policy(ckpt, device)


def _load_simple_policy(ckpt, device):
    """Load the simple MLP policy."""
    import torch.nn as nn

    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]

    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    model = SimplePolicy(state_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "type": "simple_mlp",
        "model": model,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "ckpt": ckpt,
    }


def _load_diffusion_policy(ckpt, device):
    """Load the diffusion U-Net policy."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
        DiffusionUnetLowdimPolicy,
    )

    cfg = ckpt["config"]
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_iters"],
        beta_schedule=cfg["beta_schedule"],
        clip_sample=True,
        prediction_type="epsilon",
    )

    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * cfg["n_obs_steps"],
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=cfg["down_dims"],
        kernel_size=cfg["kernel_size"],
        n_groups=cfg["n_groups"],
        cond_predict_scale=cfg["cond_predict_scale"],
    )

    policy = DiffusionUnetLowdimPolicy(
        model=unet,
        noise_scheduler=noise_scheduler,
        horizon=cfg["horizon"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        num_inference_steps=cfg["num_inference_iters"],
        obs_as_global_cond=True,
        pred_action_steps_only=False,
    )

    policy.load_state_dict(ckpt["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Diffusion U-Net: {num_params:,} params ({num_params / 1e6:.1f}M)")
    print(f"  Horizon: {cfg['horizon']}, Action steps: {cfg['n_action_steps']}, Obs steps: {cfg['n_obs_steps']}")

    return {
        "type": "diffusion_unet",
        "model": policy,
        "state_dim": obs_dim,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "n_obs_steps": cfg["n_obs_steps"],
        "n_action_steps": cfg["n_action_steps"],
        "ckpt": ckpt,
    }


class LiveHandleAugmenter:
    """
    Computes handle augmented features (11 dims) from the live MuJoCo sim,
    matching what 05b_augment_handle_data.py bakes into the training data.
    """

    OPEN_THRESHOLD = 0.90

    def __init__(self, env):
        model = env.sim.model
        self._model = model
        self.handle_bodies = []
        self.door_joints = []
        self.handle_to_joint_map = {}

        ep_meta = env.get_ep_meta()
        fixture_refs = ep_meta.get("fixture_refs", {})
        self.fixture_name = fixture_refs.get("fxtr")
        if not self.fixture_name:
            return

        for i in range(model.nbody):
            name = model.body(i).name
            if self.fixture_name in name and "handle" in name:
                self.handle_bodies.append(name)

        for i in range(model.njnt):
            name = model.joint(i).name
            if self.fixture_name in name and "door" in name:
                self.door_joints.append((name, i))

        if len(self.handle_bodies) <= 1 or len(self.door_joints) <= 1:
            self.handle_to_joint_map = {hb: self.door_joints for hb in self.handle_bodies}
        else:
            for hb in self.handle_bodies:
                hb_l = hb.lower()
                if "left" in hb_l:
                    matched = [(j, i) for j, i in self.door_joints if "left" in j.lower()]
                elif "right" in hb_l:
                    matched = [(j, i) for j, i in self.door_joints if "right" in j.lower()]
                else:
                    matched = []
                self.handle_to_joint_map[hb] = matched if matched else self.door_joints

    def _door_openness(self, data, joints):
        if not joints:
            return 0.0
        vals = []
        for _, jidx in joints:
            addr = self._model.joint(jidx).qposadr[0]
            qpos = data.qpos[addr]
            jmin, jmax = self._model.jnt_range[jidx]
            if jmax - jmin > 1e-8:
                norm = abs(qpos - jmin) / (jmax - jmin) if abs(jmin) < abs(jmax) else abs(qpos - jmax) / (jmax - jmin)
            else:
                norm = 0.0
            vals.append(np.clip(norm, 0.0, 1.0))
        return float(np.mean(vals))

    def _hinge_direction(self, handle_body):
        joints = self.handle_to_joint_map.get(handle_body, [])
        if not joints:
            return 0.0
        _, jidx = joints[0]
        jmin, jmax = self._model.jnt_range[jidx]
        return 1.0 if abs(jmin) < abs(jmax) else -1.0

    def compute(self, env):
        if not self.handle_bodies:
            return np.zeros(11, dtype=np.float32)
        data = env.sim.data
        eef_pos = data.body("gripper0_right_eef").xpos.copy()
        per_door = {hb: self._door_openness(data, self.handle_to_joint_map[hb]) for hb in self.handle_bodies}
        active = [hb for hb in self.handle_bodies if per_door[hb] < self.OPEN_THRESHOLD]
        candidates = active if active else self.handle_bodies
        dists = [np.linalg.norm(data.body(hb).xpos - eef_pos) for hb in candidates]
        target = candidates[int(np.argmin(dists))]
        handle_pos = data.body(target).xpos.copy().astype(np.float32)
        handle_to_eef = (handle_pos - eef_pos).astype(np.float32)
        openness = np.array([per_door[target]], dtype=np.float32)
        xmat = data.body(target).xmat.reshape(3, 3)
        handle_xaxis = xmat[:, 0].copy().astype(np.float32)
        hinge_dir = np.array([self._hinge_direction(target)], dtype=np.float32)
        return np.concatenate([handle_pos, handle_to_eef, openness, handle_xaxis, hinge_dir])


def extract_state(obs, state_dim, augmented_features=None):
    """Extract state vector matching the exact training feature order.

    Raw state (16 dims) from robocasa LeRobot conversion:
        [robot0_base_pos(3), robot0_base_quat(4), robot0_base_to_eef_pos(3),
         robot0_base_to_eef_quat(4), robot0_gripper_qpos(2)]
    Then augmented features (11 dims) from LiveHandleAugmenter:
        [handle_pos(3), handle_to_eef(3), openness(1), xaxis(3), hinge(1)]
    """
    RAW_STATE_KEYS = [
        "robot0_base_pos",
        "robot0_base_quat",
        "robot0_base_to_eef_pos",
        "robot0_base_to_eef_quat",
        "robot0_gripper_qpos",
    ]

    parts = []
    for key in RAW_STATE_KEYS:
        if key in obs and isinstance(obs[key], np.ndarray):
            parts.append(obs[key].flatten())
    if not parts:
        return np.zeros(state_dim, dtype=np.float32)

    raw_state = np.concatenate(parts).astype(np.float32)

    if augmented_features is not None:
        raw_dim = state_dim - len(augmented_features)
        if len(raw_state) > raw_dim:
            raw_state = raw_state[:raw_dim]
        elif len(raw_state) < raw_dim:
            raw_state = np.pad(raw_state, (0, raw_dim - len(raw_state)))
        state = np.concatenate([raw_state, augmented_features])
    else:
        if len(raw_state) < state_dim:
            raw_state = np.pad(raw_state, (0, state_dim - len(raw_state)))
        elif len(raw_state) > state_dim:
            raw_state = raw_state[:state_dim]
        state = raw_state

    return state


def get_action(policy_info, obs, obs_history, action_buffer, augmenter=None, env=None):
    """
    Get the next action from either policy type.

    For MLP: single forward pass per step.
    For diffusion: runs inference when action_buffer is empty, then
    pops from the buffer (action chunking).
    """
    import torch

    state_dim = policy_info["state_dim"]
    aug_feats = augmenter.compute(env) if (augmenter and augmenter.handle_bodies and env) else None
    state = extract_state(obs, state_dim, augmented_features=aug_feats)

    if policy_info["type"] == "simple_mlp":
        model = policy_info["model"]
        device = next(model.parameters()).device
        with torch.no_grad():
            action = model(
                torch.from_numpy(state).unsqueeze(0).to(device)
            ).cpu().numpy().squeeze(0)
        return action

    # Diffusion policy with action chunking
    model = policy_info["model"]
    device = model.device
    n_obs = policy_info["n_obs_steps"]

    obs_history.append(state)

    if len(action_buffer) == 0:
        # Need to re-plan: run diffusion inference
        while len(obs_history) < n_obs:
            obs_history.appendleft(obs_history[0])

        obs_seq = np.stack(list(obs_history), axis=0)  # (n_obs, obs_dim)
        obs_tensor = (
            torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
        )  # (1, n_obs, obs_dim)

        with torch.no_grad():
            result = model.predict_action({"obs": obs_tensor})
            action_chunk = result["action"].cpu().numpy().squeeze(0)

        for a in action_chunk:
            action_buffer.append(a)

    return action_buffer.popleft()


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(policy_info, args):
    """
    Run the policy with an interactive MuJoCo viewer window.

    The viewer opens automatically; you can pan/zoom/rotate the camera
    with the mouse while the robot executes the policy.
    """
    action_dim = policy_info["action_dim"]

    env = robosuite.make(
        env_name="OpenCabinet",
        robots="PandaOmron",
        controller_configs=load_composite_controller_config(robot="PandaOmron"),
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="robot0_frontview",
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    env = VisualizationWrapper(env)

    successes = 0
    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")
        print(f"  Running for up to {args.max_steps} steps...")
        print(f"  (Watch the viewer window — use mouse to orbit the camera)\n")

        success = False
        hold_count = 0
        obs_history = collections.deque(maxlen=policy_info.get("n_obs_steps", 1))
        action_buffer = collections.deque()
        augmenter = LiveHandleAugmenter(env)

        for step in range(args.max_steps):
            action = get_action(policy_info, obs, obs_history, action_buffer,
                                augmenter=augmenter, env=env)

            # Reorder action from LeRobot/parquet format to env/HDF5 format
            env_action = np.zeros(12, dtype=np.float32)
            env_action[0:3] = action[5:8]    # eef_position
            env_action[3:6] = action[8:11]   # eef_rotation
            env_action[6:7] = action[11:12]  # gripper_close
            env_action[7:11] = action[0:4]   # base_motion
            env_action[11:12] = action[4:5]  # control_mode

            env_dim = env.action_dim
            if len(env_action) < env_dim:
                env_action = np.pad(env_action, (0, env_dim - len(env_action)))
            elif len(env_action) > env_dim:
                env_action = env_action[:env_dim]

            obs, reward, done, info = env.step(env_action)

            # Print a brief status every 20 steps
            if step % 20 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                act_mag = float(np.abs(action).mean())
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  "
                    f"action_mag={act_mag:.3f}  [{status}]"
                )

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            # Pace the rollout so it is easy to watch
            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"\n  Result: {result}")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with video ────────────────────────────────────────────

def run_offscreen(policy_info, args):
    """
    Run the policy headlessly and save a side-by-side annotated video.

    Each frame shows the robot from the front-view camera; per-step
    diagnostics (step count, reward, success flag) are printed to the
    terminal.
    """
    import imageio
    from robocasa.utils.env_utils import create_env

    video_dir = os.path.dirname(args.video_path)
    if video_dir:
        os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768

    successes = 0
    all_frames = []  # collect frames across episodes

    for ep in range(args.num_episodes):
        print(f"\n--- Episode {ep + 1}/{args.num_episodes} ---")
        env = create_env(
            env_name="OpenCabinet",
            render_onscreen=False,
            seed=args.seed + ep,
            camera_widths=cam_w,
            camera_heights=cam_h,
        )
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")
        print(f"  Task:    {lang}")
        print(f"  Layout:  {env.layout_id}   Style: {env.style_id}")

        success = False
        hold_count = 0
        ep_frames = []
        obs_history = collections.deque(maxlen=policy_info.get("n_obs_steps", 1))
        action_buffer = collections.deque()
        augmenter = LiveHandleAugmenter(env)

        for step in range(args.max_steps):
            action = get_action(policy_info, obs, obs_history, action_buffer,
                                augmenter=augmenter, env=env)

            # Reorder action from LeRobot/parquet format to env/HDF5 format
            env_action = np.zeros(12, dtype=np.float32)
            env_action[0:3] = action[5:8]    # eef_position
            env_action[3:6] = action[8:11]   # eef_rotation
            env_action[6:7] = action[11:12]  # gripper_close
            env_action[7:11] = action[0:4]   # base_motion
            env_action[11:12] = action[4:5]  # control_mode

            env_dim = env.action_dim
            if len(env_action) < env_dim:
                env_action = np.pad(env_action, (0, env_dim - len(env_action)))
            elif len(env_action) > env_dim:
                env_action = env_action[:env_dim]

            obs, reward, done, info = env.step(env_action)

            # Render from the agent view camera
            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]  # MuJoCo renders upside-down
            ep_frames.append(frame)

            if step % 20 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                print(
                    f"  step {step:4d}  reward={reward:+.3f}  [{status}]"
                )

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "did not open cabinet"
        print(f"  Result: {result}  ({len(ep_frames)} frames)")
        if success:
            successes += 1

        all_frames.extend(ep_frames)
        env.close()

    # Write video
    print(f"\nWriting {len(all_frames)} frames to {args.video_path} ...")
    with imageio.get_writer(args.video_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video_path}")

    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/tmp/cabinet_policy_checkpoints/best_policy.pt",
        help="Path to policy checkpoint (.pt) saved by 06_train_policy.py",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="Render to video file instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/tmp/policy_rollout.mp4",
        help="Output video path (used with --offscreen)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frames per second for the saved video",
    )
    parser.add_argument(
        "--max_fr",
        type=int,
        default=20,
        help="On-screen playback rate cap (frames/second); lower = slower",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment layout/style selection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    # Load policy
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required.  Run: pip install torch")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_info = load_policy(args.checkpoint, device)
    ckpt = policy_info["ckpt"]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Type:  {policy_info['type']}")
    print(f"  Epoch {ckpt['epoch']}, loss {ckpt['loss']:.6f}")
    print(f"  State dim: {policy_info['state_dim']},  Action dim: {policy_info['action_dim']}")
    print(f"  Device: {device}")
    print()

    mode = "off-screen (video)" if args.offscreen else "on-screen (viewer window)"
    print(f"Mode:     {mode}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps/ep: {args.max_steps}")
    if args.offscreen:
        print(f"Output:   {args.video_path}")
    print()

    if args.offscreen:
        run_offscreen(policy_info, args)
    else:
        print("Opening viewer window...")
        print("  Tip: orbit the camera with the mouse to see the gripper.\n")
        run_onscreen(policy_info, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
