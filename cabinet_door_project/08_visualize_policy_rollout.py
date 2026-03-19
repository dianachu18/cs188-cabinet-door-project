"""
Step 8: Visualize a Policy Rollout
=====================================
Loads a trained policy checkpoint from 06_train_policy.py and runs it
live in the OpenCabinet environment so you can watch the robot.

Supports both simple MLP and diffusion U-Net policies (auto-detected
from checkpoint). This script is fully self-contained — it does NOT
require the diffusion_policy package.

Two rendering modes:
  On-screen  (default)  — interactive MuJoCo viewer window, real-time
  Off-screen (--offscreen) — renders to video file(s), works without display

Usage:
    # Watch live in a window (use mjpython on Mac)
    mjpython 08_visualize_policy_rollout.py --checkpoint best_diffusion_policy.pt

    # Save all 20 episodes as separate videos
    python 08_visualize_policy_rollout.py --checkpoint best_diffusion_policy.pt --offscreen

    # Save to a specific directory
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen --video_dir ./videos

    # Custom episode count
    python 08_visualize_policy_rollout.py --checkpoint ... --offscreen --num_episodes 5
"""

import os
import sys

# ── Rendering mode detection ────────────────────────────────────────────────
_OFFSCREEN = "--offscreen" in sys.argv

if _OFFSCREEN:
    if sys.platform == "linux":
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
else:
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
import math
import time

import numpy as np
import robocasa  # noqa: F401
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

# =====================================================================
#  Self-contained diffusion_policy modules (no package install needed)
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import reduce
from einops.layers.torch import Rearrange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        self.params_dict = params_dict if params_dict is not None else nn.ParameterDict()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value):
            if len(keys) == 1:
                dest[keys[0]] = value
                return
            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        out = nn.ParameterDict()
        for key, value in state_dict.items():
            if key.startswith(prefix + "params_dict"):
                param_keys = key[len(prefix + "params_dict"):].split(".")[1:]
                dfs_add(out, param_keys, value.clone())
        self.params_dict = out
        self.params_dict.requires_grad_(False)


def _fit(data, last_n_dims=1, dtype=torch.float32, mode="limits",
         output_max=1., output_min=-1., range_eps=1e-4, fit_offset=True):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype:
        data = data.type(dtype)
    dim = int(np.prod(data.shape[-last_n_dims:])) if last_n_dims > 0 else 1
    data = data.reshape(-1, dim)
    input_min, _ = data.min(0)
    input_max, _ = data.max(0)
    input_mean = data.mean(0)
    input_std = data.std(0)
    if mode == "limits" and fit_offset:
        r = input_max - input_min
        ign = r < range_eps
        r[ign] = output_max - output_min
        scale = (output_max - output_min) / r
        offset = output_min - scale * input_min
        offset[ign] = (output_max + output_min) / 2 - input_min[ign]
    else:
        scale = torch.ones_like(input_mean)
        offset = torch.zeros_like(input_mean)
    p = nn.ParameterDict({
        "scale": scale, "offset": offset,
        "input_stats": nn.ParameterDict({
            "min": input_min, "max": input_max,
            "mean": input_mean, "std": input_std,
        }),
    })
    for x in p.parameters():
        x.requires_grad_(False)
    return p


def _normalize(x, params, forward=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    s, o = params["scale"], params["offset"]
    x = x.to(device=s.device, dtype=s.dtype)
    sh = x.shape
    x = x.reshape(-1, s.shape[0])
    x = x * s + o if forward else (x - o) / s
    return x.reshape(sh)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    def normalize(self, x):
        return _normalize(x, self.params_dict, True)

    def unnormalize(self, x):
        return _normalize(x, self.params_dict, False)


class LinearNormalizer(DictOfTensorMixin):
    @torch.no_grad()
    def fit(self, data, **kw):
        if isinstance(data, dict):
            for k, v in data.items():
                self.params_dict[k] = _fit(v, **kw)
        else:
            self.params_dict["_default"] = _fit(data, **kw)

    def __getitem__(self, k):
        return SingleFieldLinearNormalizer(self.params_dict[k])

    def normalize(self, x):
        if isinstance(x, dict):
            return {k: _normalize(v, self.params_dict[k], True) for k, v in x.items()}
        return _normalize(x, self.params_dict["_default"], True)

    def unnormalize(self, x):
        if isinstance(x, dict):
            return {k: _normalize(v, self.params_dict[k], False) for k, v in x.items()}
        return _normalize(x, self.params_dict["_default"], False)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, ks, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, ks, padding=ks // 2),
            nn.GroupNorm(n_groups, out),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        h = self.dim // 2
        e = torch.exp(torch.arange(h, device=x.device) * -(math.log(10000) / (h - 1)))
        e = x[:, None] * e[None, :]
        return torch.cat((e.sin(), e.cos()), -1)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, ic, oc, cond_dim, kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(ic, oc, kernel_size, n_groups),
            Conv1dBlock(oc, oc, kernel_size, n_groups),
        ])
        cc = oc * 2 if cond_predict_scale else oc
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = oc
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cc), Rearrange("b t -> b t 1")
        )
        self.residual_conv = nn.Conv1d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        emb = self.cond_encoder(cond)
        if self.cond_predict_scale:
            emb = emb.reshape(emb.shape[0], 2, self.out_channels, 1)
            out = emb[:, 0] * out + emb[:, 1]
        else:
            out = out + emb
        return self.blocks[1](out) + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, local_cond_dim=None, global_cond_dim=None,
                 diffusion_step_embed_dim=256, down_dims=[256, 512, 1024],
                 kernel_size=3, n_groups=8, cond_predict_scale=False):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4),
            nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        kw = dict(kernel_size=kernel_size, n_groups=n_groups,
                  cond_predict_scale=cond_predict_scale)
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, d = in_out[0]
            self.local_cond_encoder = nn.ModuleList([
                ConditionalResidualBlock1D(local_cond_dim, d, cond_dim, **kw),
                ConditionalResidualBlock1D(local_cond_dim, d, cond_dim, **kw),
            ])
        mid = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid, mid, cond_dim, **kw),
            ConditionalResidualBlock1D(mid, mid, cond_dim, **kw),
        ])
        self.down_modules = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(di, do, cond_dim, **kw),
                ConditionalResidualBlock1D(do, do, cond_dim, **kw),
                Downsample1d(do) if i < len(in_out) - 1 else nn.Identity(),
            ]))
        self.up_modules = nn.ModuleList()
        for i, (di, do) in enumerate(reversed(in_out[1:])):
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(do * 2, di, cond_dim, **kw),
                ConditionalResidualBlock1D(di, di, cond_dim, **kw),
                Upsample1d(di) if i < len(in_out) - 1 else nn.Identity(),
            ]))
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        sample = einops.rearrange(sample, "b h t -> b t h")
        ts = timestep if torch.is_tensor(timestep) else torch.tensor(
            [timestep], dtype=torch.long, device=sample.device)
        if len(ts.shape) == 0:
            ts = ts[None]
        ts = ts.expand(sample.shape[0])
        gf = self.diffusion_step_encoder(ts)
        if global_cond is not None:
            gf = torch.cat([gf, global_cond], -1)
        hl = []
        if local_cond is not None:
            lc = einops.rearrange(local_cond, "b h t -> b t h")
            hl = [self.local_cond_encoder[0](lc, gf),
                  self.local_cond_encoder[1](lc, gf)]
        x, h = sample, []
        for i, (r1, r2, ds) in enumerate(self.down_modules):
            x = r1(x, gf)
            if i == 0 and hl:
                x = x + hl[0]
            x = r2(x, gf)
            h.append(x)
            x = ds(x)
        for m in self.mid_modules:
            x = m(x, gf)
        for i, (r1, r2, us) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), 1)
            x = r1(x, gf)
            if i == len(self.up_modules) and hl:
                x = x + hl[1]
            x = r2(x, gf)
            x = us(x)
        return einops.rearrange(self.final_conv(x), "b t h -> b h t")


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self, action_dim, obs_dim, max_n_obs_steps=2,
                 fix_obs_steps=True, action_visible=False):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        dev = self.device
        B, T, D = shape
        m = torch.zeros(shape, dtype=torch.bool, device=dev)
        ia = m.clone()
        ia[..., :self.action_dim] = True
        io = ~ia
        os_ = (torch.full((B,), self.max_n_obs_steps, device=dev)
               if self.fix_obs_steps
               else torch.randint(1, self.max_n_obs_steps + 1, (B,), device=dev))
        steps = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        return (steps.T < os_).T.unsqueeze(-1).expand(B, T, D) & io


class DiffusionUnetLowdimPolicy(ModuleAttrMixin):
    def __init__(self, model, noise_scheduler, horizon, obs_dim, action_dim,
                 n_action_steps, n_obs_steps, num_inference_steps=None,
                 obs_as_global_cond=True, pred_action_steps_only=False, **kwargs):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps, fix_obs_steps=True, action_visible=False)
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps)
        self.kwargs = kwargs

    def set_normalizer(self, n):
        self.normalizer.load_state_dict(n.state_dict())

    def conditional_sample(self, cd, cm, local_cond=None, global_cond=None,
                           generator=None, **kw):
        traj = torch.randn_like(cd)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            traj[cm] = cd[cm]
            out = self.model(traj, t, local_cond=local_cond, global_cond=global_cond)
            traj = self.noise_scheduler.step(out, t, traj, generator=generator, **kw).prev_sample
        traj[cm] = cd[cm]
        return traj

    def predict_action(self, obs_dict):
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        Da = self.action_dim
        dev, dt = self.device, self.dtype
        gc = nobs[:, :To].reshape(B, -1)
        shape = (B, self.horizon, Da)
        cd = torch.zeros(shape, device=dev, dtype=dt)
        cm = torch.zeros_like(cd, dtype=torch.bool)
        ns = self.conditional_sample(cd, cm, global_cond=gc, **self.kwargs)
        ap = self.normalizer["action"].unnormalize(ns[..., :Da])
        return {"action": ap[:, To:To + self.n_action_steps], "action_pred": ap}

    def compute_loss(self, batch):
        nb = self.normalizer.normalize(batch)
        obs, action = nb["obs"], nb["action"]
        gc = obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)
        traj = action
        cm = (torch.zeros_like(traj, dtype=torch.bool)
              if self.pred_action_steps_only
              else self.mask_generator(traj.shape))
        noise = torch.randn_like(traj)
        ts = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                           (traj.shape[0],), device=traj.device).long()
        noisy = self.noise_scheduler.add_noise(traj, noise, ts)
        noisy[cm] = traj[cm]
        pred = self.model(noisy, ts, global_cond=gc)
        target = noise if self.noise_scheduler.config.prediction_type == "epsilon" else traj
        loss = F.mse_loss(pred, target, reduction="none") * (~cm).float()
        return reduce(loss, "b ... -> b (...)", "mean").mean()


# =====================================================================
#  Policy loading (supports both MLP and diffusion)
# =====================================================================

def load_policy(checkpoint_path, device):
    """Load a policy checkpoint (MLP or Diffusion U-Net, auto-detected)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy_type = ckpt.get("policy_type", "simple_mlp")

    if policy_type == "diffusion_unet":
        return _load_diffusion_policy(ckpt, device)
    else:
        return _load_simple_policy(ckpt, device)


def _load_simple_policy(ckpt, device):
    """Load the simple MLP policy."""
    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]

    class SimplePolicy(nn.Module):
        def __init__(self, sd, ad, hidden_dim=256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(sd, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, ad), nn.Tanh(),
            )

        def forward(self, state):
            return self.net(state)

    model = SimplePolicy(state_dim, action_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {
        "type": "simple_mlp", "model": model,
        "state_dim": state_dim, "action_dim": action_dim, "ckpt": ckpt,
    }


def _load_diffusion_policy(ckpt, device):
    """Load the diffusion U-Net policy (self-contained, no package needed)."""
    cfg = ckpt["config"]
    obs_dim = ckpt["obs_dim"]
    action_dim = ckpt["action_dim"]

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_iters"],
        beta_schedule=cfg["beta_schedule"],
        clip_sample=True, prediction_type="epsilon",
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
        model=unet, noise_scheduler=noise_scheduler,
        horizon=cfg["horizon"], obs_dim=obs_dim, action_dim=action_dim,
        n_action_steps=cfg["n_action_steps"], n_obs_steps=cfg["n_obs_steps"],
        num_inference_steps=cfg["num_inference_iters"],
        obs_as_global_cond=True, pred_action_steps_only=False,
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy = policy.to(device)
    policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"  Diffusion U-Net: {num_params:,} params ({num_params / 1e6:.1f}M)")
    print(f"  Horizon: {cfg['horizon']}, Action steps: {cfg['n_action_steps']}, Obs steps: {cfg['n_obs_steps']}")

    return {
        "type": "diffusion_unet", "model": policy,
        "state_dim": obs_dim, "action_dim": action_dim,
        "obs_dim": obs_dim, "n_obs_steps": cfg["n_obs_steps"],
        "n_action_steps": cfg["n_action_steps"], "ckpt": ckpt,
    }


# =====================================================================
#  Live handle augmentation & state extraction
# =====================================================================

class LiveHandleAugmenter:
    """Computes handle augmented features (11 dims) from the live MuJoCo sim."""

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
                norm = (abs(qpos - jmin) / (jmax - jmin) if abs(jmin) < abs(jmax)
                        else abs(qpos - jmax) / (jmax - jmin))
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
        per_door = {hb: self._door_openness(data, self.handle_to_joint_map[hb])
                    for hb in self.handle_bodies}
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


def get_door_joint_states(env):
    """Read all door joint positions from the MuJoCo sim directly."""
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
                if abs(jmin) < abs(jmax):
                    norm = abs(qpos - jmin) / (jmax - jmin)
                else:
                    norm = abs(qpos - jmax) / (jmax - jmin)
                result[name] = float(np.clip(norm, 0.0, 1.0))
            else:
                result[name] = 0.0
    return result


def extract_state(obs, state_dim, augmented_features=None):
    """Extract state vector matching the exact training feature order."""
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
    """Get the next action from either policy type."""
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

    model = policy_info["model"]
    device = model.device
    n_obs = policy_info["n_obs_steps"]
    obs_history.append(state)

    if len(action_buffer) == 0:
        while len(obs_history) < n_obs:
            obs_history.appendleft(obs_history[0])
        obs_seq = np.stack(list(obs_history), axis=0)
        obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(device)
        with torch.no_grad():
            result = model.predict_action({"obs": obs_tensor})
            action_chunk = result["action"].cpu().numpy().squeeze(0)
        for a in action_chunk:
            action_buffer.append(a)

    return action_buffer.popleft()


def reorder_action(action):
    """Reorder action from LeRobot/parquet format to env/HDF5 format."""
    env_action = np.zeros(12, dtype=np.float32)
    env_action[0:3] = action[5:8]    # eef_position
    env_action[3:6] = action[8:11]   # eef_rotation
    env_action[6:7] = action[11:12]  # gripper_close
    env_action[7:11] = action[0:4]   # base_motion
    env_action[11:12] = action[4:5]  # control_mode
    return env_action


def pad_action(env_action, env_dim):
    """Pad or trim action to match environment action dimension."""
    if len(env_action) < env_dim:
        return np.pad(env_action, (0, env_dim - len(env_action)))
    elif len(env_action) > env_dim:
        return env_action[:env_dim]
    return env_action


# ── On-screen rollout ────────────────────────────────────────────────────────

def run_onscreen(policy_info, args):
    """Run the policy with an interactive MuJoCo viewer window."""
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

        success = False
        hold_count = 0
        max_openness = 0.0
        obs_history = collections.deque(maxlen=policy_info.get("n_obs_steps", 1))
        action_buffer = collections.deque()
        augmenter = LiveHandleAugmenter(env)

        for step in range(args.max_steps):
            action = get_action(policy_info, obs, obs_history, action_buffer,
                                augmenter=augmenter, env=env)
            env_action = pad_action(reorder_action(action), env.action_dim)
            obs, reward, done, info = env.step(env_action)

            door_states = get_door_joint_states(env)
            if door_states:
                max_openness = max(max_openness, max(door_states.values()))

            if step % 20 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                print(f"  step {step:4d}  reward={reward:+.3f}  "
                      f"door={max_openness:.1%}  [{status}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

            time.sleep(1.0 / args.max_fr)

        result = "SUCCESS" if success else "FAIL"
        print(f"  Result: {result}  (max door openness: {max_openness:.1%})")
        if success:
            successes += 1

    env.close()
    print(f"\nFinal: {successes}/{args.num_episodes} episodes succeeded.")


# ── Off-screen rollout with per-episode videos ────────────────────────────────

def run_offscreen(policy_info, args):
    """
    Run the policy headlessly and save a separate video for each episode.

    Videos are saved as:
        <video_dir>/episode_01.mp4
        <video_dir>/episode_02.mp4
        ...
    Plus a combined video with all episodes concatenated.
    """
    import imageio
    from robocasa.utils.env_utils import create_env

    video_dir = args.video_dir
    os.makedirs(video_dir, exist_ok=True)

    cam_h, cam_w = 512, 768

    successes = 0
    all_frames = []
    results_summary = []

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
        max_openness = 0.0
        ep_frames = []
        obs_history = collections.deque(maxlen=policy_info.get("n_obs_steps", 1))
        action_buffer = collections.deque()
        augmenter = LiveHandleAugmenter(env)

        for step in range(args.max_steps):
            action = get_action(policy_info, obs, obs_history, action_buffer,
                                augmenter=augmenter, env=env)
            env_action = pad_action(reorder_action(action), env.action_dim)
            obs, reward, done, info = env.step(env_action)

            # Render frame
            frame = env.sim.render(
                height=cam_h, width=cam_w, camera_name="robot0_agentview_center"
            )[::-1]
            ep_frames.append(frame)

            # Track door openness
            door_states = get_door_joint_states(env)
            if door_states:
                max_openness = max(max_openness, max(door_states.values()))

            if step % 50 == 0:
                checking = env._check_success()
                status = "cabinet OPEN" if checking else "in progress"
                door_str = f"{max_openness:.1%}" if door_states else "n/a"
                print(f"  step {step:4d}  door={door_str}  [{status}]")

            if env._check_success():
                hold_count += 1
                if hold_count >= 15:
                    success = True
                    break
            else:
                hold_count = 0

        result = "SUCCESS" if success else "FAIL"
        print(f"  Result: {result}  (max door: {max_openness:.1%}, {len(ep_frames)} frames)")
        if success:
            successes += 1

        results_summary.append({
            "episode": ep + 1,
            "result": result,
            "steps": len(ep_frames),
            "max_door": max_openness,
            "layout": env.layout_id,
            "style": env.style_id,
            "task": lang,
        })

        # Save per-episode video
        ep_video_path = os.path.join(video_dir, f"episode_{ep + 1:02d}.mp4")
        with imageio.get_writer(ep_video_path, fps=args.fps) as writer:
            for frame in ep_frames:
                writer.append_data(frame)
        print(f"  Saved: {ep_video_path}")

        all_frames.extend(ep_frames)
        env.close()

    # Save combined video with all episodes
    combined_path = os.path.join(video_dir, "all_episodes.mp4")
    print(f"\nWriting combined video ({len(all_frames)} frames) to {combined_path} ...")
    with imageio.get_writer(combined_path, fps=args.fps) as writer:
        for frame in all_frames:
            writer.append_data(frame)
    print(f"Combined video saved: {combined_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"  Results Summary: {successes}/{args.num_episodes} succeeded")
    print(f"{'=' * 70}")
    print(f"  {'Ep':>3}  {'Result':>7}  {'Steps':>5}  {'Max Door':>8}  {'Layout':>6}  {'Style':>5}  Task")
    print(f"  {'---':>3}  {'------':>7}  {'-----':>5}  {'--------':>8}  {'------':>6}  {'-----':>5}  ----")
    for r in results_summary:
        print(f"  {r['episode']:3d}  {r['result']:>7}  {r['steps']:5d}  "
              f"{r['max_door']:>7.1%}  {r['layout']:6d}  {r['style']:5d}  {r['task']}")
    print()
    print(f"  Videos saved to: {video_dir}/")
    print(f"  Individual:  episode_01.mp4 ... episode_{args.num_episodes:02d}.mp4")
    print(f"  Combined:    all_episodes.mp4")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize a trained policy rollout in OpenCabinet"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="/tmp/cabinet_diffusion_checkpoints/best_diffusion_policy.pt",
        help="Path to policy checkpoint (.pt)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Number of episodes to run (default: 20)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--offscreen", action="store_true",
        help="Render to video file(s) instead of opening a viewer window",
    )
    parser.add_argument(
        "--video_dir", type=str, default="./rollout_videos",
        help="Directory to save episode videos (used with --offscreen)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Frames per second for saved videos",
    )
    parser.add_argument(
        "--max_fr", type=int, default=20,
        help="On-screen playback rate cap (frames/second)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for environment layout/style selection",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Rollout Visualizer")
    print("=" * 60)
    print()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Train a policy first with:  python 06_train_policy.py --diffusion")
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
        print(f"Video dir: {args.video_dir}")
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
