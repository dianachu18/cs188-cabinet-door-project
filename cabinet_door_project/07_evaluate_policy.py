"""
Step 7: Evaluate a Trained Policy
===================================
Runs a trained policy in the OpenCabinet environment and reports
success rate across multiple episodes and kitchen scenes.

Supports both:
  - Simple MLP policy (from 06_train_policy.py)
  - Diffusion U-Net policy (from 06_train_policy.py --diffusion)

Usage:
    # Evaluate the simple BC policy from Step 6
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_policy_checkpoints/best_policy.pt

    # Evaluate diffusion policy (auto-detected from checkpoint)
    python 07_evaluate_policy.py --checkpoint /tmp/cabinet_diffusion_checkpoints/best_diffusion_policy.pt

    # Evaluate with more episodes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --num_rollouts 50

    # Evaluate on target (held-out) kitchen scenes
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --split target

    # Save evaluation videos
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --video_path /tmp/eval_videos.mp4

    # Require ALL doors open for success (strict mode)
    python 07_evaluate_policy.py --checkpoint path/to/policy.pt --no_any_door_open

For evaluating official Diffusion Policy / pi-0 / GR00T checkpoints,
use the evaluation scripts from those repos instead (see 06_train_policy.py).
"""

import argparse
import collections
import os
import sys

# Force osmesa (CPU offscreen renderer) on Linux/WSL2 -- EGL requires
# /dev/dri device access that is unavailable in WSL environments.
if sys.platform == "linux":
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# Add diffusion_policy to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DP_ROOT = os.path.join(SCRIPT_DIR, "diffusion_policy")
if DP_ROOT not in sys.path:
    sys.path.insert(0, DP_ROOT)

import numpy as np

import robocasa  # noqa: F401
from robocasa.utils.env_utils import create_env


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def load_policy(checkpoint_path, device):
    """Load a trained policy checkpoint (MLP or Diffusion)."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy_type = checkpoint.get("policy_type", "simple_mlp")

    if policy_type == "diffusion_unet":
        return _load_diffusion_policy(checkpoint, device)
    else:
        return _load_simple_policy(checkpoint, device)


def _load_simple_policy(checkpoint, device):
    """Load the simple MLP policy from 06_train_policy.py."""
    import torch.nn as nn

    state_dim = checkpoint["state_dim"]
    action_dim = checkpoint["action_dim"]

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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded MLP policy from: {checkpoint.get('_path', '?')}")
    print(f"  Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")

    return {
        "type": "simple_mlp",
        "model": model,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def _load_diffusion_policy(checkpoint, device):
    """Load the diffusion U-Net policy from 09_train_lowdim_unet.py."""
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
    from diffusion_policy.policy.diffusion_unet_lowdim_policy import (
        DiffusionUnetLowdimPolicy,
    )

    cfg = checkpoint["config"]
    obs_dim = checkpoint["obs_dim"]
    action_dim = checkpoint["action_dim"]

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_iters"],
        beta_schedule=cfg["beta_schedule"],
        clip_sample=True,
        prediction_type="epsilon",
    )

    global_cond_dim = obs_dim * cfg["n_obs_steps"]

    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
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

    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Loaded Diffusion U-Net policy")
    print(f"  Trained for {checkpoint['epoch']} epochs, loss={checkpoint['loss']:.6f}")
    print(f"  Obs dim: {obs_dim}, Action dim: {action_dim}")
    print(f"  Horizon: {cfg['horizon']}, Action steps: {cfg['n_action_steps']}, Obs steps: {cfg['n_obs_steps']}")
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    return {
        "type": "diffusion_unet",
        "model": policy,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "state_dim": obs_dim,
        "n_obs_steps": cfg["n_obs_steps"],
        "n_action_steps": cfg["n_action_steps"],
    }


class LiveHandleAugmenter:
    """
    Computes handle augmented features (11 dims) from the live MuJoCo sim,
    matching what 05b_augment_handle_data.py bakes into the training data.

    Features (per timestep):
        handle_pos          (3)  Handle 3D world position
        handle_to_eef_pos   (3)  Handle position relative to end-effector
        door_openness       (1)  Normalized door joint state [0=closed, 1=open]
        handle_xaxis        (3)  Handle body x-axis (door facing direction)
        hinge_direction     (1)  Hinge direction indicator (+1 right, -1 left)
    """

    OPEN_THRESHOLD = 0.90

    def __init__(self, env):
        """Call after env.reset() each episode."""
        model = env.sim.model
        self._model = model
        self.handle_bodies = []
        self.door_joints = []
        self.handle_to_joint_map = {}

        # Get fixture name from episode metadata
        ep_meta = env.get_ep_meta()
        fixture_refs = ep_meta.get("fixture_refs", {})
        self.fixture_name = fixture_refs.get("fxtr")

        if not self.fixture_name:
            return

        # Find handle bodies
        for i in range(model.nbody):
            name = model.body(i).name
            if self.fixture_name in name and "handle" in name:
                self.handle_bodies.append(name)

        # Find door joints
        for i in range(model.njnt):
            name = model.joint(i).name
            if self.fixture_name in name and "door" in name:
                self.door_joints.append((name, i))

        # Map handles → joints
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
                if abs(jmin) < abs(jmax):
                    norm = abs(qpos - jmin) / (jmax - jmin)
                else:
                    norm = abs(qpos - jmax) / (jmax - jmin)
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
        """Return 11-dim augmented feature vector from live sim state."""
        if not self.handle_bodies:
            return np.zeros(11, dtype=np.float32)

        data = env.sim.data
        eef_pos = data.body("gripper0_right_eef").xpos.copy()

        # Per-handle door openness
        per_door = {
            hb: self._door_openness(data, self.handle_to_joint_map[hb])
            for hb in self.handle_bodies
        }

        # Pick nearest handle whose door is not yet fully open
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
    """Extract a fixed-size state vector from observations.

    MUST match the exact feature order used during training:
        [robot0_base_pos(3), robot0_base_quat(4), robot0_base_to_eef_pos(3),
         robot0_base_to_eef_quat(4), robot0_gripper_qpos(2)]  = 16 raw dims
    Then augmented features (11 dims) from LiveHandleAugmenter:
        [handle_pos(3), handle_to_eef(3), openness(1), xaxis(3), hinge(1)]
    """
    # Extract raw state in the EXACT order used by robocasa's LeRobot conversion
    RAW_STATE_KEYS = [
        "robot0_base_pos",          # 3 dims
        "robot0_base_quat",         # 4 dims
        "robot0_base_to_eef_pos",   # 3 dims
        "robot0_base_to_eef_quat",  # 4 dims
        "robot0_gripper_qpos",      # 2 dims
    ]

    state_parts = []
    for key in RAW_STATE_KEYS:
        if key in obs and isinstance(obs[key], np.ndarray):
            state_parts.append(obs[key].flatten())

    if not state_parts:
        return np.zeros(state_dim, dtype=np.float32)

    raw_state = np.concatenate(state_parts).astype(np.float32)

    if augmented_features is not None:
        # Pad/truncate raw to exactly (state_dim - aug_dim)
        raw_dim = state_dim - len(augmented_features)
        if len(raw_state) > raw_dim:
            raw_state = raw_state[:raw_dim]
        elif len(raw_state) < raw_dim:
            raw_state = np.pad(raw_state, (0, raw_dim - len(raw_state)))
        state = np.concatenate([raw_state, augmented_features])
    else:
        state = raw_state
        if len(state) < state_dim:
            state = np.pad(state, (0, state_dim - len(state)))
        elif len(state) > state_dim:
            state = state[:state_dim]

    return state


def get_door_joint_states(env):
    """Read all door joint positions from the MuJoCo sim directly.

    Returns dict of {joint_name: normalized_openness} where 0=closed, 1=fully open.
    """
    model = env.sim.model
    data = env.sim.data
    ep_meta = env.get_ep_meta()
    fxtr_name = ep_meta.get("fixture_refs", {}).get("fxtr", "")
    if not fxtr_name:
        return {}

    result = {}
    for i in range(model.njnt):
        name = model.joint(i).name
        if fxtr_name in name and "door" in name:
            jmin, jmax = model.jnt_range[i]
            addr = model.joint(i).qposadr[0]
            qpos = data.qpos[addr]
            if jmax - jmin > 1e-8:
                # Normalize: 0 = closed, 1 = fully open
                if abs(jmin) < abs(jmax):
                    norm = abs(qpos - jmin) / (jmax - jmin)
                else:
                    norm = abs(qpos - jmax) / (jmax - jmin)
                result[name] = float(np.clip(norm, 0.0, 1.0))
            else:
                result[name] = 0.0
    return result


def check_any_door_open(env, threshold=0.90):
    """Check if ANY door of the fixture is open (relaxed success criterion)."""
    joint_states = get_door_joint_states(env)
    if joint_states:
        for norm_qpos in joint_states.values():
            if norm_qpos >= threshold:
                return True
        return False
    # Fall back to default check
    return env._check_success()


def run_evaluation(
    policy_info,
    num_rollouts,
    max_steps,
    split,
    video_path,
    seed,
    any_door_open,
    door_threshold=0.90,
):
    """Run evaluation rollouts and collect statistics."""
    import torch
    import imageio

    policy_type = policy_info["type"]
    state_dim = policy_info["state_dim"]
    action_dim = policy_info["action_dim"]

    if policy_type == "simple_mlp":
        model = policy_info["model"]
        device = next(model.parameters()).device
    else:
        model = policy_info["model"]
        device = model.device

    env = create_env(
        env_name="OpenCabinet",
        render_onscreen=False,
        seed=seed,
        split=split,
        camera_widths=256,
        camera_heights=256,
    )

    video_writer = None
    if video_path:
        os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)

    results = {
        "successes": [],
        "episode_lengths": [],
        "rewards": [],
        "max_door_openness": [],
    }

    for ep in range(num_rollouts):
        obs = env.reset()
        ep_meta = env.get_ep_meta()
        lang = ep_meta.get("lang", "")

        ep_reward = 0.0
        success = False
        max_openness = 0.0

        # Live augmentation (handle features) for diffusion policies
        augmenter = LiveHandleAugmenter(env)

        # Debug: print augmenter info on first episode
        # if ep == 0:
        #     aug_feats_debug = augmenter.compute(env) if augmenter.handle_bodies else None
        #     raw_state = extract_state(obs, state_dim)
        #     full_state = extract_state(obs, state_dim, augmented_features=aug_feats_debug)
        #     print(f"  [DEBUG] Handle bodies found: {len(augmenter.handle_bodies)}")
        #     print(f"  [DEBUG] Raw state dims: {np.count_nonzero(raw_state)} non-zero of {len(raw_state)}")
        #     if aug_feats_debug is not None:
        #         print(f"  [DEBUG] Augmented features (11): {aug_feats_debug}")
        #     print(f"  [DEBUG] Full state (first 5): {full_state[:5]}")
        #     print(f"  [DEBUG] Full state (last 5):  {full_state[-5:]}")

        # For diffusion policy: action chunk buffer and obs history
        action_buffer = collections.deque()
        obs_history = collections.deque(maxlen=policy_info.get("n_obs_steps", 1))

        for step in range(max_steps):
            aug_feats = augmenter.compute(env) if augmenter.handle_bodies else None
            state = extract_state(obs, state_dim, augmented_features=aug_feats)

            if policy_type == "simple_mlp":
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = model(state_tensor).cpu().numpy().squeeze(0)
            else:
                # Diffusion policy with action chunking
                obs_history.append(state)

                if len(action_buffer) == 0:
                    # Need to re-plan: run diffusion inference
                    n_obs = policy_info["n_obs_steps"]

                    # Pad obs history if not enough steps yet
                    while len(obs_history) < n_obs:
                        obs_history.appendleft(obs_history[0])

                    obs_seq = np.stack(list(obs_history), axis=0)  # (n_obs, obs_dim)
                    obs_tensor = (
                        torch.from_numpy(obs_seq)
                        .float()
                        .unsqueeze(0)
                        .to(device)
                    )  # (1, n_obs, obs_dim)

                    with torch.no_grad():
                        result = model.predict_action({"obs": obs_tensor})
                        action_chunk = result["action"].cpu().numpy().squeeze(0)
                        # action_chunk: (n_action_steps, action_dim)

                    for a in action_chunk:
                        action_buffer.append(a)

                action = action_buffer.popleft()

            # Reorder action from LeRobot/parquet format to env/HDF5 format.
            # Model predicts: [base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]
            # Env expects:    [eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]
            env_action = np.zeros(12, dtype=np.float32)
            env_action[0:3] = action[5:8]    # eef_position
            env_action[3:6] = action[8:11]   # eef_rotation
            env_action[6:7] = action[11:12]  # gripper_close
            env_action[7:11] = action[0:4]   # base_motion
            env_action[11:12] = action[4:5]  # control_mode

            # Pad/truncate to match environment action dim if needed
            env_action_dim = env.action_dim
            if len(env_action) < env_action_dim:
                env_action = np.pad(env_action, (0, env_action_dim - len(env_action)))
            elif len(env_action) > env_action_dim:
                env_action = env_action[:env_action_dim]

            obs, reward, done, info = env.step(env_action)
            ep_reward += reward

            if video_writer is not None:
                frame = env.sim.render(
                    height=512, width=768, camera_name="robot0_agentview_center"
                )[::-1]
                video_writer.append_data(frame)

            # Track max door openness
            door_states = get_door_joint_states(env)
            if door_states:
                cur_max = max(door_states.values())
                max_openness = max(max_openness, cur_max)

            # Check success
            if any_door_open:
                success = check_any_door_open(env, threshold=door_threshold)
            else:
                success = env._check_success()

            if success:
                break

        results["successes"].append(success)
        results["episode_lengths"].append(step + 1)
        results["rewards"].append(ep_reward)
        results["max_door_openness"].append(max_openness)

        status = "SUCCESS" if success else "FAIL"
        print(
            f"  Episode {ep + 1:3d}/{num_rollouts}: {status:7s} "
            f"(steps={step + 1:4d}, reward={ep_reward:.1f}, max_door={max_openness:.1%}) "
            f'layout={env.layout_id}, style={env.style_id}, task="{lang}"'
        )

    if video_writer:
        video_writer.close()

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained OpenCabinet policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to policy checkpoint (.pt file)",
    )
    parser.add_argument(
        "--num_rollouts", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max_steps", type=int, default=500, help="Max steps per episode"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="pretrain",
        choices=["pretrain", "target"],
        help="Kitchen scene split to evaluate on",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to save evaluation video (optional)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--any_door_open",
        action="store_true",
        default=True,
        help="Consider one door open as success (default: True)",
    )
    parser.add_argument(
        "--no_any_door_open",
        action="store_true",
        help="Require ALL doors open for success (strict mode)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Door openness threshold for success (0-1, default 0.90)",
    )
    args = parser.parse_args()

    any_door_open = args.any_door_open and not args.no_any_door_open

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)

    print("=" * 60)
    print("  OpenCabinet - Policy Evaluation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Success criterion: {'any door open' if any_door_open else 'all doors open (strict)'}")

    # Load the trained policy
    policy_info = load_policy(args.checkpoint, device)

    # Run evaluation
    print_section(f"Evaluating on {args.split} split ({args.num_rollouts} episodes)")

    print(f"Door open threshold: {args.threshold:.0%}")

    results = run_evaluation(
        policy_info=policy_info,
        num_rollouts=args.num_rollouts,
        max_steps=args.max_steps,
        split=args.split,
        video_path=args.video_path,
        seed=args.seed,
        any_door_open=any_door_open,
        door_threshold=args.threshold,
    )

    # Print summary
    print_section("Evaluation Results")

    num_success = sum(results["successes"])
    success_rate = num_success / args.num_rollouts * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["rewards"])

    print(f"  Policy type:    {policy_info['type']}")
    print(f"  Split:          {args.split}")
    print(f"  Episodes:       {args.num_rollouts}")
    print(f"  Success crit:   {'any door' if any_door_open else 'all doors'}")
    print(f"  Successes:      {num_success}/{args.num_rollouts}")
    print(f"  Door threshold: {args.threshold:.0%}")
    print(f"  Success rate:   {success_rate:.1f}%")
    print(f"  Avg ep length:  {avg_length:.1f} steps")
    print(f"  Avg reward:     {avg_reward:.3f}")
    if results["max_door_openness"]:
        avg_max_door = np.mean(results["max_door_openness"])
        best_max_door = max(results["max_door_openness"])
        print(f"  Avg max door:   {avg_max_door:.1%}")
        print(f"  Best max door:  {best_max_door:.1%}")

    if args.video_path:
        print(f"\n  Video saved to: {args.video_path}")

    # Context for expected performance
    print_section("Performance Context")
    print(
        "Expected success rates from the RoboCasa benchmark:\n"
        "\n"
        "  Method            | Pretrain | Target\n"
        "  ------------------|----------|-------\n"
        "  Random actions    |    ~0%   |   ~0%\n"
        "  Diffusion Policy  |  ~30-60% | ~20-50%\n"
        "  pi-0              |  ~40-70% | ~30-60%\n"
        "  GR00T N1.5        |  ~35-65% | ~25-55%\n"
        "\n"
        "Note: The simple MLP policy from Step 6 is not expected to\n"
        "achieve meaningful success rates. Use the diffusion policy\n"
        "(06_train_policy.py --diffusion) for real results."
    )


if __name__ == "__main__":
    main()
