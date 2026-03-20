# Diffusion Policy for Robotic Cabinet Door Opening

### CS 188 — Introduction to Robotics | UCLA | Final Project

A diffusion-based imitation learning system that trains a robot to open kitchen cabinet doors in the [RoboCasa](https://robocasa.ai/) simulation environment. The policy uses a **1D Convolutional U-Net** with **action chunking** and **DDPM noise scheduling** to learn from 133 human demonstrations, achieving **55% success rate** on the OpenCabinet task.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
  - [Step 0–2: Environment Setup & Exploration](#step-0-2-environment-setup--exploration)
  - [Step 3–4: Data Collection](#step-3-4-data-collection)
  - [Step 5: Data Augmentation](#step-5-data-augmentation)
  - [Step 6: Training](#step-6-training)
  - [Step 7: Evaluation](#step-7-evaluation)
  - [Step 8: Visualization](#step-8-visualization)
  - [Step 9: DAgger (Interactive Improvement)](#step-9-dagger-interactive-improvement)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Task:** A PandaOmron mobile manipulator (7-DOF Franka Panda arm + Omron wheeled base) must locate and pull open a hinged cabinet door across diverse procedurally generated kitchen scenes.

**Approach:** We train a **Diffusion Policy** — a denoising diffusion probabilistic model (DDPM) that generates action sequences conditioned on robot observations. Instead of predicting a single action per timestep, the policy predicts a *horizon* of 16 future actions and executes them in short chunks, enabling smooth and temporally coherent manipulation behavior.

**Key features:**
- 1D Convolutional U-Net (`ConditionalUnet1D`) with ~43M parameters
- Action chunking: predict 16 steps, execute 8
- Live handle augmentation: 11 extra observation dims (handle position, door openness, hinge direction)
- EMA weight averaging (decay=0.995) for stable training
- Cosine beta schedule (`squaredcos_cap_v2`) for diffusion noise

---

## Architecture

```
Observations (27-dim)                    Actions (12-dim x 16 horizon)
─────────────────────                    ────────────────────────────
 Proprioception (16):                     End-effector delta pos (3)
   gripper_qpos (2)                       End-effector delta rot (3)
   base_pos (3)                           Gripper open/close (1)
   base_quat (4)                          Base motion (4)
   base_to_eef_pos (3)                    Control mode (1)
   base_to_eef_quat (4)
                           ┌───────────────────────────┐
 Handle features (11):     │   ConditionalUnet1D       │
   handle_pos (3)          │   (1D Conv, ~43M params)  │
   handle_to_eef_pos (3)   │                           │
   door_openness (1)  ───▶ │   Encoder: [128,256,512]  │ ───▶ Denoised actions
   handle_xaxis (3)        │   Kernel size: 3          │      (12-dim x 16 steps)
   hinge_direction (1)     │   Groups: 8               │
                           │   Diffusion steps: 100    │
                           │   Inference steps: 16     │
                           └───────────────────────────┘

                    DDPM Training                    Inference
                    ─────────────                    ─────────
                    1. Sample action chunk            1. Start from random noise
                    2. Add noise (step t)             2. Denoise 16 iterations
                    3. Predict noise ε_θ              3. Execute first 2 actions
                    4. MSE loss on noise              4. Re-observe, repeat
```

---

## Results

| Metric | Value |
|--------|-------|
| Success rate (90% door threshold) | **50%** |
| Best episode door openness | **91.4%** |
| Average max door openness | ~52.3% |
| Model parameters | 43.0M |
| Training epochs | 500 |
| Training loss (best) | 0.002275 |
| Training time | ~180 min on 5080 GPU |

The robot successfully learns to:
1. Navigate toward the cabinet
2. Locate the door handle using augmented features
3. Grasp and pull the handle to open the door

---

## Installation

### Prerequisites
- Python 3.10+
- macOS or Linux/WSL
- ~10 GB disk space for kitchen assets

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd cs188-cabinet-door-project

# Run the install script
./install.sh

# Activate the virtual environment
source .venv/bin/activate

# Verify installation
cd cabinet_door_project
python 00_verify_installation.py
```

The install script will:
- Create a Python virtual environment (`.venv`)
- Clone and install robosuite and robocasa
- Install all Python dependencies (PyTorch, numpy, matplotlib, etc.)
- Download RoboCasa kitchen assets (~10 GB)

> **macOS note:** Scripts that open a rendering window (03, 05, 08 on-screen mode) require `mjpython` instead of `python`.

---

## Project Structure

```
cs188-cabinet-door-project/
├── README.md
├── install.sh                              # Installation script
├── pyproject.toml
├── main.py
│
├── cabinet_door_project/
│   ├── 00_verify_installation.py           # Verify MuJoCo + RoboCasa setup
│   ├── 01_explore_environment.py           # Inspect observation/action spaces
│   ├── 02_random_rollouts.py               # Random agent baseline + video
│   ├── 03_teleop_collect_demos.py          # Keyboard teleoperation for demos
│   ├── 04_download_dataset.py              # Download OpenCabinet demonstrations
│   ├── 05_playback_demonstrations.py       # Replay expert demonstrations
│   ├── 05b_augment_handle_data.py          # Add handle features to dataset
│   ├── 06_train_policy.py                  # Train diffusion policy (local)
│   ├── 07_evaluate_policy.py               # Evaluate policy + door openness
│   ├── 08_visualize_policy_rollout.py      # Record per-episode rollout videos
│   ├── copy_dagger.py                      # Copy DAgger episodes into training set
│   │
│   ├── configs/
│   │   └── diffusion_policy.yaml           # Training hyperparameters
│   ├── diffusion_policy/                   # ConditionalUnet1D, normalizer, etc.
│   ├── train_diffusion_colab.ipynb         # Google Colab training notebook
│   └── notebook.ipynb                      # Interactive exploration notebook
│
├── robocasa/                               # RoboCasa simulation framework
└── robosuite/                              # Robot control backend
```

---

## Pipeline

### Step 0–2: Environment Setup & Exploration

```bash
python 00_verify_installation.py      # Check MuJoCo, robosuite, RoboCasa
python 01_explore_environment.py      # Print obs/action space details
python 02_random_rollouts.py          # Random agent → /tmp/cabinet_random_rollouts.mp4
```

### Step 3–4: Data Collection

```bash
# Collect your own demos via keyboard teleoperation (Mac: use mjpython)
mjpython 03_teleop_collect_demos.py

# Or download pre-collected 50-episode OpenCabinet dataset
python 04_download_dataset.py
```

**Keyboard controls for teleoperation:**

| Key | Action |
|-----|--------|
| `Ctrl+q` | Reset simulation |
| `spacebar` | Toggle gripper |
| `↑ → ↓ ←` | Move in x-y plane |
| `. ;` | Move vertically |
| `o p` | Rotate (yaw) |
| `y h` | Rotate (pitch) |
| `e r` | Rotate (roll) |
| `b` | Toggle arm/base mode |

### Step 5: Data Augmentation

```bash
python 05_playback_demonstrations.py    # Visualize expert demos
python 05b_augment_handle_data.py       # Add 11-dim handle features
```

`05b` replays saved MuJoCo states to extract runtime features not stored in the original parquet files:

| Feature | Dims | Description |
|---------|------|-------------|
| `handle_pos` | 3 | Handle 3D world position |
| `handle_to_eef_pos` | 3 | Handle relative to end-effector |
| `door_openness` | 1 | Normalized joint state (0=closed, 1=open) |
| `handle_xaxis` | 3 | Door facing direction |
| `hinge_direction` | 1 | Hinge side indicator (+1 right, -1 left) |

### Step 6: Training

**Local training:**
```bash
# Diffusion policy (recommended)
python 06_train_policy.py --diffusion

# With custom hyperparameters
python 06_train_policy.py --diffusion --epochs 350 --batch_size 128

# Quick test with small model
python 06_train_policy.py --diffusion --fast

# Simple MLP baseline (educational only)
python 06_train_policy.py
```


**Default hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Horizon | 16 |
| Observation steps | 2 |
| Action steps (executed) | 8 |
| Diffusion iterations (train) | 100 |
| Diffusion iterations (inference) | 16 |
| Beta schedule | `squaredcos_cap_v2` |
| U-Net channels | [128, 256, 512] |
| Kernel size | 3 |
| Epochs | 500 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| EMA decay | 0.995 |

Checkpoints are saved to `/tmp/cabinet_diffusion_checkpoints/`.

### Step 7: Evaluation

```bash
# Basic evaluation (20 episodes)
python 07_evaluate_policy.py \
    --checkpoint /tmp/cabinet_diffusion_checkpoints/best_diffusion_policy.pt

# Relaxed door-open threshold
python 07_evaluate_policy.py \
    --checkpoint best_diffusion_policy.pt \
    --threshold 0.30

# More episodes, save video
python 07_evaluate_policy.py \
    --checkpoint best_diffusion_policy.pt \
    --num_rollouts 20 \
    --threshold 0.30 \
    --video_path /tmp/eval.mp4
```

The evaluator reads door joint states directly from MuJoCo simulation data, tracking `max_door_openness` per episode and reporting average/best values.

### Step 8: Visualization

```bash
# Save per-episode videos (off-screen, works without display)
python 08_visualize_policy_rollout.py \
    --checkpoint best_diffusion_policy.pt \
    --offscreen

# Custom settings
python 08_visualize_policy_rollout.py \
    --checkpoint best_diffusion_policy.pt \
    --offscreen \
    --num_episodes 5 \
    --video_dir ./my_videos

# Live interactive viewer (Mac: use mjpython)
mjpython 08_visualize_policy_rollout.py \
    --checkpoint best_diffusion_policy.pt
```

Saves individual episode videos (`episode_01.mp4`, etc.) and a combined `all_episodes.mp4` to `--video_dir` (default: `./rollout_videos/`).

### Step 9: DAgger (Interactive Improvement)

**DAgger (Dataset Aggregation)** is an iterative imitation learning technique that improves policy robustness by collecting human corrections on the trained policy's failures.

**How it works:**
1. The trained policy drives the robot autonomously
2. When the policy makes mistakes, the human operator intervenes with keyboard corrections
3. These corrective trajectories are saved as new training data
4. The policy is retrained on the combined dataset (original demos + DAgger corrections)

**Collecting DAgger episodes:**
```bash
# Run the policy with DAgger mode enabled (Mac: use mjpython)
python 03_teleop_collect_demos.py \
    --dagger \
    --checkpoint checkpoints/best_diffusion_policy.pt
```

The policy will execute autonomously while you observe. Use the same keyboard controls as regular teleoperation to intervene when the robot struggles (e.g., missing the handle, moving in the wrong direction). DAgger episodes are saved to `data/dagger/chunk-000/`.

**Adding DAgger data to the training set:**

DAgger episodes must be copied into the augmented training directory with renumbered filenames to avoid overwriting existing episodes:

```bash
python copy_dagger.py
```

This copies the collected DAgger parquet files (e.g., `episode_000000.parquet` → `episode_000107.parquet`) into the augmented dataset directory alongside the original 107 expert demonstrations.

**Retraining with DAgger data:**
```bash
python 06_train_policy.py --diffusion --epochs 500
```

The training script automatically picks up all parquet files in the augmented directory, including the newly added DAgger episodes.

**Important considerations:**
- DAgger records **all** timesteps (both policy-driven and human-corrected). Since the policy drives most of the episode, the DAgger data contains a mix of policy actions and expert corrections. A large number of DAgger episodes relative to expert demos can dilute the training signal.
- Quality over quantity: fewer high-quality correction episodes targeting specific failure modes are more valuable than many full-length episodes.
- We collected 26 DAgger episodes to supplement the 107 original expert demonstrations.

---

## Key Design Decisions

### Why Diffusion Policy over vanilla BC?
Standard behavior cloning with MSE loss averages over multimodal demonstrations (e.g., approaching the handle from the left vs. right), producing ineffective mean actions. Diffusion models naturally handle multimodality by learning the full action distribution.

### Why Action Chunking?
Single-step action prediction produces jerky, temporally incoherent behavior. Predicting a horizon of 16 actions and executing 2 at a time (then replanning) yields smooth trajectories while maintaining reactivity to the environment.

### Why Handle Augmentation?
The raw proprioceptive observations (16 dims) don't tell the robot where the cabinet handle is. The 11-dim handle features from `05b` provide the critical spatial relationship between the end-effector and the grasp target.

### Action Reordering
LeRobot parquet files store actions in a different order than what `robosuite` expects. The evaluation and visualization scripts include `reorder_action()` to handle this mismatch — a critical detail for correct execution.

### Door State Monitoring
The default `env.fxtr.get_joint_state()` API was unreliable. We implemented `get_door_joint_states()` to read door hinge joint positions directly from MuJoCo's `sim.data.qpos`, providing accurate normalized door openness tracking.

---

## Observation & Action Spaces

### Observation Space (27-dim)

| Component | Dims | Source |
|-----------|------|--------|
| `gripper_qpos` | 2 | Gripper finger positions |
| `base_pos` | 3 | Mobile base position (x, y, z) |
| `base_quat` | 4 | Mobile base orientation |
| `base_to_eef_pos` | 3 | End-effector position relative to base |
| `base_to_eef_quat` | 4 | End-effector orientation relative to base |
| `handle_pos` | 3 | Cabinet handle world position (augmented) |
| `handle_to_eef_pos` | 3 | Handle-to-EEF vector (augmented) |
| `door_openness` | 1 | Normalized door joint state (augmented) |
| `handle_xaxis` | 3 | Door facing direction (augmented) |
| `hinge_direction` | 1 | Hinge side: +1 or -1 (augmented) |

### Action Space (12-dim)

| Component | Dims | Description |
|-----------|------|-------------|
| End-effector delta position | 3 | (dx, dy, dz) |
| End-effector delta rotation | 3 | Axis-angle |
| Gripper | 1 | 0=open, 1=close |
| Base motion | 4 | (forward, lateral, yaw, torso) |
| Control mode | 1 | 0=arm, 1=base |

---

## Software Stack

```
┌──────────────────────────────────────────┐
│          Diffusion Policy (ours)         │
│  ConditionalUnet1D + DDPM + Normalizer   │
├──────────────────────────────────────────┤
│              RoboCasa                    │
│  Kitchen scenes, OpenCabinet task logic  │
│  Fixture management, 2500+ layouts       │
├──────────────────────────────────────────┤
│              robosuite                   │
│  Robot models (PandaOmron), controllers  │
│  Observation/action framework            │
├──────────────────────────────────────────┤
│          MuJoCo 3.3.1 (Physics)          │
│  Contact dynamics, rendering, sensors    │
└──────────────────────────────────────────┘
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `MuJoCo version must be 3.3.1` | `pip install mujoco==3.3.1` |
| `numpy version must be 2.2.5` | `pip install numpy==2.2.5` |
| Rendering crashes on Mac | Use `mjpython` instead of `python` |
| `GLFW error` on headless server | `export MUJOCO_GL=egl` or `osmesa` |
| Out of GPU memory | Reduce `batch_size` (try 64 or 32) |
| Kitchen assets not found | `python -m robocasa.scripts.download_kitchen_assets` |
| `doors: n/a` in eval output | Update `07_evaluate_policy.py` to use `get_door_joint_states()` |
| `TypeError: missing positional argument 'joint_names'` | Use direct MuJoCo joint reading instead of `env.fxtr.get_joint_state()` |
| LeRobot action ordering mismatch | Ensure `reorder_action()` is applied before `env.step()` |

---

## Acknowledgments

This project builds on the following open-source work:

- **[RoboCasa](https://robocasa.ai/)** — Large-scale simulation benchmark for everyday robot tasks
- **[robosuite](https://robosuite.ai/)** — Robot learning simulation framework
- **[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)** (Chi et al., 2023) — Visuomotor policy learning via action diffusion
- **[LeRobot](https://github.com/huggingface/lerobot)** (Hugging Face) — Robot learning dataset format and tools
- **[MuJoCo](https://mujoco.readthedocs.io/)** — Physics simulation engine
- **CS 188 Starter Code** by Holden GS (holdengs @ cs.ucla.edu)

### GenAI Disclaimer

Claude (Anthropic) was used via Claude Code CLI to assist with: debugging environment API issues, hyperparameter tuning guidance, and code modifications (door state monitoring, video recording). All training experiments, data collection, architectural decisions, and evaluations were performed by the team. 