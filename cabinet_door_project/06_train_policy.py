"""
Step 6: Train a Policy for OpenCabinet
========================================
Trains either a simple MLP behavior-cloning baseline or a diffusion policy
using a 1D Convolutional U-Net with action chunking.

The MLP baseline illustrates the data-loading -> training -> checkpoint
pipeline but will not solve the task. The diffusion policy (--diffusion flag)
uses a self-contained ConditionalUnet1D (~18M params) and can actually learn
to open cabinet doors.

This script is fully self-contained — it does NOT require the diffusion_policy
package to be installed. All necessary modules are defined inline.

Prerequisites (for diffusion mode):
    python 04_download_dataset.py      # Download demonstrations
    python 05b_augment_handle_data.py  # Augment with handle features

Usage:
    # Simple MLP baseline (educational)
    python 06_train_policy.py [--epochs 50] [--batch_size 32] [--lr 1e-4]

    # Diffusion policy with 1D U-Net (recommended)
    python 06_train_policy.py --diffusion
    python 06_train_policy.py --diffusion --epochs 350 --batch_size 128

    # Quick local sanity check (small model, few epochs)
    python 06_train_policy.py --diffusion --fast

    # Print instructions for official external repos
    python 06_train_policy.py --use_diffusion_policy
"""

import argparse
import copy
import math
import os
import sys
import time
import yaml
from typing import Dict, Union

import numpy as np

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


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# --- utils ---
def dict_apply(x, func):
    return {
        k: dict_apply(v, func) if isinstance(v, dict) else func(v)
        for k, v in x.items()
    }


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
        self.params_dict = (
            params_dict if params_dict is not None else nn.ParameterDict()
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
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
                param_keys = key[len(prefix + "params_dict") :].split(".")[1:]
                dfs_add(out, param_keys, value.clone())
        self.params_dict = out
        self.params_dict.requires_grad_(False)


# --- Normalizer ---
def _fit(
    data,
    last_n_dims=1,
    dtype=torch.float32,
    mode="limits",
    output_max=1.0,
    output_min=-1.0,
    range_eps=1e-4,
    fit_offset=True,
):
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
    p = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
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
            return {
                k: _normalize(v, self.params_dict[k], True) for k, v in x.items()
            }
        return _normalize(x, self.params_dict["_default"], True)

    def unnormalize(self, x):
        if isinstance(x, dict):
            return {
                k: _normalize(v, self.params_dict[k], False) for k, v in x.items()
            }
        return _normalize(x, self.params_dict["_default"], False)


# --- Conv1D components ---
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
        e = torch.exp(
            torch.arange(h, device=x.device) * -(math.log(10000) / (h - 1))
        )
        e = x[:, None] * e[None, :]
        return torch.cat((e.sin(), e.cos()), -1)


# --- Conditional Residual Block ---
class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        ic,
        oc,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(ic, oc, kernel_size, n_groups),
                Conv1dBlock(oc, oc, kernel_size, n_groups),
            ]
        )
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


# --- Conditional U-Net 1D ---
class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        kw = dict(
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )
        self.local_cond_encoder = None
        if local_cond_dim is not None:
            _, d = in_out[0]
            self.local_cond_encoder = nn.ModuleList(
                [
                    ConditionalResidualBlock1D(local_cond_dim, d, cond_dim, **kw),
                    ConditionalResidualBlock1D(local_cond_dim, d, cond_dim, **kw),
                ]
            )
        mid = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(mid, mid, cond_dim, **kw),
                ConditionalResidualBlock1D(mid, mid, cond_dim, **kw),
            ]
        )
        self.down_modules = nn.ModuleList()
        for i, (di, do) in enumerate(in_out):
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(di, do, cond_dim, **kw),
                        ConditionalResidualBlock1D(do, do, cond_dim, **kw),
                        Downsample1d(do) if i < len(in_out) - 1 else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList()
        for i, (di, do) in enumerate(reversed(in_out[1:])):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(do * 2, di, cond_dim, **kw),
                        ConditionalResidualBlock1D(di, di, cond_dim, **kw),
                        Upsample1d(di) if i < len(in_out) - 1 else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        sample = einops.rearrange(sample, "b h t -> b t h")
        ts = (
            timestep
            if torch.is_tensor(timestep)
            else torch.tensor([timestep], dtype=torch.long, device=sample.device)
        )
        if len(ts.shape) == 0:
            ts = ts[None]
        ts = ts.expand(sample.shape[0])
        gf = self.diffusion_step_encoder(ts)
        if global_cond is not None:
            gf = torch.cat([gf, global_cond], -1)
        hl = []
        if local_cond is not None:
            lc = einops.rearrange(local_cond, "b h t -> b t h")
            hl = [
                self.local_cond_encoder[0](lc, gf),
                self.local_cond_encoder[1](lc, gf),
            ]
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


# --- Mask Generator ---
class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(
        self,
        action_dim,
        obs_dim,
        max_n_obs_steps=2,
        fix_obs_steps=True,
        action_visible=False,
    ):
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
        ia[..., : self.action_dim] = True
        io = ~ia
        os_ = (
            torch.full((B,), self.max_n_obs_steps, device=dev)
            if self.fix_obs_steps
            else torch.randint(1, self.max_n_obs_steps + 1, (B,), device=dev)
        )
        steps = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        return (steps.T < os_).T.unsqueeze(-1).expand(B, T, D) & io


# --- Diffusion Policy ---
class DiffusionUnetLowdimPolicy(ModuleAttrMixin):
    def __init__(
        self,
        model,
        noise_scheduler,
        horizon,
        obs_dim,
        action_dim,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        pred_action_steps_only=False,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.num_inference_steps = (
            num_inference_steps or noise_scheduler.config.num_train_timesteps
        )
        self.kwargs = kwargs

    def set_normalizer(self, n):
        self.normalizer.load_state_dict(n.state_dict())

    def conditional_sample(
        self, cd, cm, local_cond=None, global_cond=None, generator=None, **kw
    ):
        traj = torch.randn_like(cd)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            traj[cm] = cd[cm]
            out = self.model(
                traj, t, local_cond=local_cond, global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                out, t, traj, generator=generator, **kw
            ).prev_sample
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
        return {
            "action": ap[:, To : To + self.n_action_steps],
            "action_pred": ap,
        }

    def compute_loss(self, batch):
        nb = self.normalizer.normalize(batch)
        obs, action = nb["obs"], nb["action"]
        gc = obs[:, : self.n_obs_steps, :].reshape(obs.shape[0], -1)
        traj = action
        cm = (
            torch.zeros_like(traj, dtype=torch.bool)
            if self.pred_action_steps_only
            else self.mask_generator(traj.shape)
        )
        noise = torch.randn_like(traj)
        ts = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (traj.shape[0],),
            device=traj.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(traj, noise, ts)
        noisy[cm] = traj[cm]
        pred = self.model(noisy, ts, global_cond=gc)
        target = (
            noise
            if self.noise_scheduler.config.prediction_type == "epsilon"
            else traj
        )
        loss = F.mse_loss(pred, target, reduction="none") * (~cm).float()
        return reduce(loss, "b ... -> b (...)", "mean").mean()


# =====================================================================
#  Config helpers
# =====================================================================

def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_path():
    """Get the path to the OpenCabinet dataset."""
    import robocasa  # noqa: F401
    from robocasa.utils.dataset_registry_utils import get_ds_path

    path = get_ds_path("OpenCabinet", source="human")
    if path is None or not os.path.exists(path):
        print("ERROR: Dataset not found. Run 04_download_dataset.py first.")
        sys.exit(1)
    return path


# =====================================================================
#  Simple MLP Policy (baseline)
# =====================================================================

def train_simple_policy(config):
    """
    Train a simple behavior-cloning policy.

    This is a simplified training loop to illustrate the pipeline.
    For real results, use --diffusion mode.
    """
    from torch.utils.data import DataLoader, Dataset

    print_section("Simple Behavior Cloning Policy")

    dataset_path = get_dataset_path()
    print(f"Dataset: {dataset_path}")

    # ----------------------------------------------------------------
    # 1. Build a simple dataset from the LeRobot format
    # ----------------------------------------------------------------
    print("\nLoading dataset...")

    class CabinetDemoDataset(Dataset):
        def __init__(self, dataset_path, max_episodes=None):
            import pyarrow.parquet as pq

            self.states = []
            self.actions = []

            data_dir = os.path.join(dataset_path, "data")
            if not os.path.exists(data_dir):
                data_dir = os.path.join(dataset_path, "lerobot", "data")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(
                    f"Data directory not found under: {dataset_path}\n"
                    "Make sure you downloaded the dataset with 04_download_dataset.py"
                )

            chunk_dir = os.path.join(data_dir, "chunk-000")
            if not os.path.exists(chunk_dir):
                raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

            parquet_files = sorted(
                f for f in os.listdir(chunk_dir) if f.endswith(".parquet")
            )
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files found in {chunk_dir}"
                )

            episodes_loaded = 0
            for pf in parquet_files:
                table = pq.read_table(os.path.join(chunk_dir, pf))
                df = table.to_pandas()

                state_cols = [
                    c for c in df.columns if c.startswith("observation.state")
                ]
                action_cols = [
                    c
                    for c in df.columns
                    if c == "action" or c.startswith("action.")
                ]

                if not state_cols or not action_cols:
                    state_cols = [
                        c
                        for c in df.columns
                        if "gripper" in c or "base" in c or "eef" in c
                    ]
                    action_cols = [c for c in df.columns if "action" in c]

                if state_cols and action_cols:
                    for _, row in df.iterrows():
                        state_parts = []
                        for c in state_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                state_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                state_parts.append(float(val))
                        action_parts = []
                        for c in action_cols:
                            val = row[c]
                            if isinstance(val, np.ndarray):
                                action_parts.extend(val.flatten().tolist())
                            elif isinstance(val, (int, float, np.floating)):
                                action_parts.append(float(val))

                        if state_parts and action_parts:
                            self.states.append(
                                np.array(state_parts, dtype=np.float32)
                            )
                            self.actions.append(
                                np.array(action_parts, dtype=np.float32)
                            )

                episodes_loaded += 1
                if max_episodes and episodes_loaded >= max_episodes:
                    break

            if len(self.states) == 0:
                print(
                    "WARNING: Could not extract state-action pairs from parquet files."
                )
                self._generate_synthetic_data()

            self.states = np.array(self.states, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.float32)

            print(f"Loaded {len(self.states)} state-action pairs")
            print(f"State dim:  {self.states.shape[-1]}")
            print(f"Action dim: {self.actions.shape[-1]}")

        def _generate_synthetic_data(self):
            rng = np.random.default_rng(42)
            for _ in range(1000):
                state = rng.standard_normal(16).astype(np.float32)
                action = rng.standard_normal(12).astype(np.float32) * 0.1
                self.states.append(state)
                self.actions.append(action)

        def __len__(self):
            return len(self.states)

        def __getitem__(self, idx):
            return (
                torch.from_numpy(self.states[idx]),
                torch.from_numpy(self.actions[idx]),
            )

    dataset = CabinetDemoDataset(dataset_path, max_episodes=50)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    # 2. Define a simple MLP policy
    # ----------------------------------------------------------------
    state_dim = dataset.states.shape[-1]
    action_dim = dataset.actions.shape[-1]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SimplePolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # ----------------------------------------------------------------
    # 3. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    print(f"Epochs:     {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"LR:         {config['learning_rate']}")

    checkpoint_dir = config.get(
        "checkpoint_dir", "/tmp/cabinet_policy_checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
    for epoch in range(config["epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        model.train()
        for states_batch, actions_batch in dataloader:
            states_batch = states_batch.to(device)
            actions_batch = actions_batch.to(device)

            pred_actions = model(states_batch)
            loss = nn.functional.mse_loss(pred_actions, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1:4d}/{config['epochs']}  Loss: {avg_loss:.6f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, "best_policy.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                },
                ckpt_path,
            )

    final_path = os.path.join(checkpoint_dir, "final_policy.pt")
    torch.save(
        {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "state_dim": state_dim,
            "action_dim": action_dim,
        },
        final_path,
    )

    print(f"\nTraining complete!")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Best checkpoint:  {ckpt_path}")
    print(f"Final checkpoint: {final_path}")

    print_section("Next Steps")
    print(
        "This simple MLP policy is for educational purposes only.\n"
        "For a policy that can actually solve the task, re-run with:\n"
        "\n"
        "  python 06_train_policy.py --diffusion\n"
    )


# =====================================================================
#  Diffusion Policy (1D U-Net with action chunking)
# =====================================================================

def get_diffusion_default_config():
    """Default hyperparameters for diffusion policy training.

    These match the Colab training notebook that produced working results
    (robot reaches handle, grasps, and pulls door open).
    """
    return {
        # Temporal parameters
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 2,  # short chunks = frequent replanning
        # Diffusion parameters
        "num_diffusion_iters": 100,
        "num_inference_iters": 16,
        "beta_schedule": "squaredcos_cap_v2",
        # U-Net architecture (~18M params)
        "down_dims": [128, 256, 512],
        "kernel_size": 5,  # wider temporal receptive field
        "n_groups": 8,
        "diffusion_step_embed_dim": 256,
        "cond_predict_scale": True,
        # Training
        "epochs": 350,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "ema_decay": 0.995,
        "lr_warmup_steps": 500,
        "grad_clip_norm": 1.0,
        # Data augmentation
        "obs_noise_std": 0.01,  # Gaussian noise on observations
        # Paths
        "checkpoint_dir": "/tmp/cabinet_diffusion_checkpoints",
    }


def build_diffusion_dataset(dataset_path, horizon, n_obs_steps, obs_noise_std=0.0):
    """Build a temporal-window dataset from parquet files for diffusion training."""
    import pyarrow.parquet as pq
    from torch.utils.data import Dataset

    # Try augmented data first, fall back to raw
    data_dir = os.path.join(dataset_path, "augmented")
    using_augmented = True
    if not os.path.exists(data_dir):
        using_augmented = False
        data_dir = os.path.join(dataset_path, "data", "chunk-000")
        if not os.path.exists(data_dir):
            data_dir = os.path.join(dataset_path, "lerobot", "data", "chunk-000")

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found under: {dataset_path}")
        sys.exit(1)

    print(f"Loading data from: {data_dir}")
    print(f"Using augmented data: {using_augmented}")

    parquet_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )
    if not parquet_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        sys.exit(1)

    # Load all episodes
    all_obs = []
    all_actions = []

    for pf in parquet_files:
        table = pq.read_table(os.path.join(data_dir, pf))
        df = table.to_pandas()

        state_cols = [c for c in df.columns if c.startswith("observation.state")]
        aug_cols = []
        for aug_name in [
            "observation.handle_pos",
            "observation.handle_to_eef_pos",
            "observation.door_openness",
            "observation.handle_xaxis",
            "observation.hinge_direction",
        ]:
            if aug_name in df.columns:
                aug_cols.append(aug_name)
        obs_cols = state_cols + aug_cols

        action_cols = [
            c for c in df.columns if c == "action" or c.startswith("action.")
        ]

        if not obs_cols or not action_cols:
            continue

        ep_obs = []
        ep_actions = []
        for _, row in df.iterrows():
            obs_parts = []
            for c in obs_cols:
                val = row[c]
                if isinstance(val, np.ndarray):
                    obs_parts.extend(val.flatten().tolist())
                elif isinstance(val, (int, float, np.floating)):
                    obs_parts.append(float(val))

            action_parts = []
            for c in action_cols:
                val = row[c]
                if isinstance(val, np.ndarray):
                    action_parts.extend(val.flatten().tolist())
                elif isinstance(val, (int, float, np.floating)):
                    action_parts.append(float(val))

            if obs_parts and action_parts:
                ep_obs.append(obs_parts)
                ep_actions.append(action_parts)

        if ep_obs:
            all_obs.append(np.array(ep_obs, dtype=np.float32))
            all_actions.append(np.array(ep_actions, dtype=np.float32))

    if not all_obs:
        print("ERROR: No valid state-action data found in parquet files.")
        sys.exit(1)

    obs_dim = all_obs[0].shape[-1]
    action_dim = all_actions[0].shape[-1]

    print(f"Loaded {len(all_obs)} episodes")
    print(
        f"Obs dim: {obs_dim} (state={len(state_cols)} cols + augmented={len(aug_cols)} cols)"
    )
    print(f"Action dim: {action_dim}")
    if obs_noise_std > 0:
        print(f"Obs noise augmentation: std={obs_noise_std}")

    # Build temporal window indices
    indices = []
    for ep_idx, ep in enumerate(all_obs):
        T = len(ep)
        for t in range(n_obs_steps - 1, T - horizon + 1):
            indices.append((ep_idx, t))

    print(f"Total training windows: {len(indices)}")

    class CabinetDiffusionDataset(Dataset):
        def __init__(self):
            self.obs = all_obs
            self.actions = all_actions
            self.indices = indices
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.horizon = horizon
            self.n_obs_steps = n_obs_steps
            self.obs_noise_std = obs_noise_std

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            ep_idx, t = self.indices[idx]
            ep_obs = self.obs[ep_idx]
            ep_act = self.actions[ep_idx]

            # Observation window: [t-n_obs+1, ..., t] -> (n_obs_steps, obs_dim)
            obs_start = t - self.n_obs_steps + 1
            obs = ep_obs[obs_start : t + 1].copy()

            # Action window: [t, ..., t+horizon-1] -> (horizon, action_dim)
            action = ep_act[t : t + self.horizon]

            # Pad if near episode end
            if len(action) < self.horizon:
                pad = np.repeat(
                    action[-1:], self.horizon - len(action), axis=0
                )
                action = np.concatenate([action, pad], axis=0)

            # Observation noise augmentation for better generalization
            if self.obs_noise_std > 0:
                obs = obs + np.random.randn(*obs.shape).astype(np.float32) * self.obs_noise_std

            return {
                "obs": torch.from_numpy(obs),
                "action": torch.from_numpy(action),
            }

    return CabinetDiffusionDataset()


def save_diffusion_checkpoint(policy, config, obs_dim, action_dim, epoch, loss, path):
    """Save a diffusion policy checkpoint with all info needed to reconstruct."""
    torch.save(
        {
            "policy_type": "diffusion_unet",
            "epoch": epoch,
            "loss": loss,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "state_dim": obs_dim,  # backward compat with eval script
            "policy_state_dict": policy.state_dict(),
            "config": {
                "horizon": config["horizon"],
                "n_obs_steps": config["n_obs_steps"],
                "n_action_steps": config["n_action_steps"],
                "num_diffusion_iters": config["num_diffusion_iters"],
                "num_inference_iters": config["num_inference_iters"],
                "beta_schedule": config["beta_schedule"],
                "down_dims": config["down_dims"],
                "kernel_size": config["kernel_size"],
                "n_groups": config["n_groups"],
                "diffusion_step_embed_dim": config["diffusion_step_embed_dim"],
                "cond_predict_scale": config["cond_predict_scale"],
            },
        },
        path,
    )


def train_diffusion_policy(config):
    """
    Train a diffusion policy with 1D U-Net and action chunking.

    Uses a self-contained ConditionalUnet1D with global observation
    conditioning and DDPM noise scheduling. No external diffusion_policy
    package required.
    """
    from torch.utils.data import DataLoader

    print_section("Diffusion Policy Training (1D U-Net)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # 1. Load dataset
    # ----------------------------------------------------------------
    dataset_path = get_dataset_path()
    dataset = build_diffusion_dataset(
        dataset_path,
        horizon=config["horizon"],
        n_obs_steps=config["n_obs_steps"],
        obs_noise_std=config.get("obs_noise_std", 0.0),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    # ----------------------------------------------------------------
    # 2. Fit normalizer
    # ----------------------------------------------------------------
    print("\nFitting normalizer...")
    all_obs_flat = np.concatenate(dataset.obs, axis=0)
    all_act_flat = np.concatenate(dataset.actions, axis=0)

    normalizer = LinearNormalizer()
    normalizer.fit(
        data={
            "obs": torch.from_numpy(all_obs_flat).float(),
            "action": torch.from_numpy(all_act_flat).float(),
        },
        mode="limits",
        output_max=1.0,
        output_min=-1.0,
    )

    # ----------------------------------------------------------------
    # 3. Build model
    # ----------------------------------------------------------------
    print("\nBuilding model...")
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule=config["beta_schedule"],
        clip_sample=True,
        prediction_type="epsilon",
    )

    global_cond_dim = obs_dim * config["n_obs_steps"]

    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=config["diffusion_step_embed_dim"],
        down_dims=config["down_dims"],
        kernel_size=config["kernel_size"],
        n_groups=config["n_groups"],
        cond_predict_scale=config["cond_predict_scale"],
    )

    policy = DiffusionUnetLowdimPolicy(
        model=unet,
        noise_scheduler=noise_scheduler,
        horizon=config["horizon"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_action_steps=config["n_action_steps"],
        n_obs_steps=config["n_obs_steps"],
        num_inference_steps=config["num_inference_iters"],
        obs_as_global_cond=True,
        pred_action_steps_only=False,
    )

    policy.set_normalizer(normalizer)
    policy = policy.to(device)

    # EMA model
    ema_policy = copy.deepcopy(policy)
    ema_policy.eval()

    num_params = sum(p.numel() for p in policy.parameters())
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    # ----------------------------------------------------------------
    # 4. Optimizer and scheduler
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    total_steps = config["epochs"] * len(dataloader)
    warmup_steps = config["lr_warmup_steps"]

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----------------------------------------------------------------
    # 5. Training loop
    # ----------------------------------------------------------------
    print_section("Training")
    print(f"Epochs:          {config['epochs']}")
    print(f"Batch size:      {config['batch_size']}")
    print(f"LR:              {config['learning_rate']}")
    print(f"Horizon:         {config['horizon']}")
    print(f"Obs steps:       {config['n_obs_steps']}")
    print(f"Action steps:    {config['n_action_steps']}")
    print(f"Diffusion iters: {config['num_diffusion_iters']} (train) / {config['num_inference_iters']} (inference)")
    print(f"EMA decay:       {config['ema_decay']}")
    print(f"U-Net dims:      {config['down_dims']}, kernel={config['kernel_size']}")
    print(f"Obs noise:       {config.get('obs_noise_std', 0.0)}")

    checkpoint_dir = config.get(
        "checkpoint_dir", "/tmp/cabinet_diffusion_checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    avg_loss = float("inf")
    ema_decay = config["ema_decay"]
    grad_clip = config.get("grad_clip_norm", 1.0)

    t0 = time.time()

    for epoch in range(config["epochs"]):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss = policy.compute_loss(batch)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()

            # EMA update
            with torch.no_grad():
                for p_ema, p in zip(
                    ema_policy.parameters(), policy.parameters()
                ):
                    p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = (time.time() - t0) / 60
            print(
                f"  Epoch {epoch + 1:4d}/{config['epochs']}  "
                f"Loss: {avg_loss:.6f}  LR: {lr:.2e}  [{elapsed:.1f}m]"
            )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_diffusion_checkpoint(
                ema_policy,
                config,
                obs_dim,
                action_dim,
                epoch,
                best_loss,
                os.path.join(checkpoint_dir, "best_diffusion_policy.pt"),
            )

        # Periodic checkpoint
        if (epoch + 1) % 100 == 0:
            save_diffusion_checkpoint(
                ema_policy,
                config,
                obs_dim,
                action_dim,
                epoch,
                avg_loss,
                os.path.join(checkpoint_dir, f"diffusion_policy_epoch{epoch + 1}.pt"),
            )

    # Final checkpoint
    save_diffusion_checkpoint(
        ema_policy,
        config,
        obs_dim,
        action_dim,
        config["epochs"],
        avg_loss,
        os.path.join(checkpoint_dir, "final_diffusion_policy.pt"),
    )

    elapsed = (time.time() - t0) / 60
    print(f"\nTraining complete! ({elapsed:.1f} min)")
    print(f"Best loss:        {best_loss:.6f}")
    print(f"Checkpoints in:   {checkpoint_dir}")

    print_section("Next Steps")
    print(
        f"Evaluate your policy:\n"
        f"  python 07_evaluate_policy.py \\\n"
        f"    --checkpoint {checkpoint_dir}/best_diffusion_policy.pt\n"
        f"\n"
        f"  # With relaxed threshold (recommended for 107-demo dataset):\n"
        f"  python 07_evaluate_policy.py \\\n"
        f"    --checkpoint {checkpoint_dir}/best_diffusion_policy.pt \\\n"
        f"    --threshold 0.30"
    )


# =====================================================================
#  External repo instructions
# =====================================================================

def print_diffusion_policy_instructions():
    """Print instructions for using the official Diffusion Policy repo."""
    print_section("Official Diffusion Policy Training")
    print(
        "For production-quality policy training, use the official repos:\n"
        "\n"
        "Option A: Diffusion Policy (recommended for single-task)\n"
        "  git clone https://github.com/robocasa-benchmark/diffusion_policy\n"
        "  cd diffusion_policy && pip install -e .\n"
        "\n"
        "  # Train\n"
        "  python train.py \\\n"
        "    --config-name=train_diffusion_transformer_bs192 \\\n"
        "    task=robocasa/OpenCabinet\n"
        "\n"
        "  # Evaluate\n"
        "  python eval_robocasa.py \\\n"
        "    --checkpoint <path-to-checkpoint> \\\n"
        "    --task_set atomic \\\n"
        "    --split target\n"
        "\n"
        "Option B: pi-0 via OpenPi (for foundation model fine-tuning)\n"
        "  git clone https://github.com/robocasa-benchmark/openpi\n"
        "  cd openpi && pip install -e . && pip install -e packages/openpi-client/\n"
        "\n"
        "  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 python scripts/train.py \\\n"
        "    robocasa_OpenCabinet --exp-name=cabinet_door\n"
        "\n"
        "Option C: GR00T N1.5 (NVIDIA foundation model)\n"
        "  git clone https://github.com/robocasa-benchmark/Isaac-GR00T\n"
        "  cd groot && pip install -e .\n"
        "\n"
        "  python scripts/gr00t_finetune.py \\\n"
        "    --output-dir experiments/cabinet_door \\\n"
        "    --dataset_soup robocasa_OpenCabinet \\\n"
        "    --max_steps 50000\n"
    )


# =====================================================================
#  Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a policy for OpenCabinet"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other args)",
    )
    parser.add_argument(
        "--diffusion",
        action="store_true",
        help="Train diffusion policy with 1D U-Net (recommended)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Prediction horizon (diffusion mode)",
    )
    parser.add_argument(
        "--n_obs_steps",
        type=int,
        default=None,
        help="Observation context steps (diffusion mode)",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Action chunk size (diffusion mode)",
    )
    parser.add_argument(
        "--obs_noise_std",
        type=float,
        default=None,
        help="Observation noise std for augmentation (default 0.01)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use smaller model and fewer epochs for quick local testing",
    )
    parser.add_argument(
        "--use_diffusion_policy",
        action="store_true",
        help="Print instructions for using the official Diffusion Policy repo",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenCabinet - Policy Training")
    print("=" * 60)

    if args.use_diffusion_policy:
        print_diffusion_policy_instructions()
        return

    if args.diffusion:
        # Diffusion policy mode
        config = get_diffusion_default_config()

        if args.config:
            yaml_config = load_config(args.config)
            config.update(
                {k: v for k, v in yaml_config.items() if v is not None}
            )

        # --fast: small model for quick CPU sanity checks
        if args.fast:
            config["down_dims"] = [64, 128, 256]
            config["diffusion_step_embed_dim"] = 128
            config["num_diffusion_iters"] = 20
            config["num_inference_iters"] = 5
            config["epochs"] = 10
            config["batch_size"] = 32
            config["lr_warmup_steps"] = 50

        # CLI overrides
        if args.epochs is not None:
            config["epochs"] = args.epochs
        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.lr is not None:
            config["learning_rate"] = args.lr
        if args.horizon is not None:
            config["horizon"] = args.horizon
        if args.n_obs_steps is not None:
            config["n_obs_steps"] = args.n_obs_steps
        if args.n_action_steps is not None:
            config["n_action_steps"] = args.n_action_steps
        if args.obs_noise_std is not None:
            config["obs_noise_std"] = args.obs_noise_std
        if args.checkpoint_dir is not None:
            config["checkpoint_dir"] = args.checkpoint_dir

        train_diffusion_policy(config)
    else:
        # Simple MLP baseline
        if args.config:
            config = load_config(args.config)
        else:
            config = {
                "epochs": args.epochs or 50,
                "batch_size": args.batch_size or 32,
                "learning_rate": args.lr or 1e-4,
                "checkpoint_dir": args.checkpoint_dir
                or "/tmp/cabinet_policy_checkpoints",
            }

        train_simple_policy(config)


if __name__ == "__main__":
    main()
