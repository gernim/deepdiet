# Experiment Tracking & Configuration Guide

This document covers the experiment tracking, batch running, and configuration management tools available in DeepDiet.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Weights & Biases Integration](#weights--biases-integration)
3. [Experiment Runner](#experiment-runner)
4. [Hydra Configuration](#hydra-configuration)
5. [GCP Workflow](#gcp-workflow)

---

## Quick Start

### First-Time Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Login to W&B (one-time)
wandb login
```

### Run a Single Experiment with W&B

```bash
python -m src.train \
    --use-side-frames \
    --use-overhead \
    --use-depth \
    --side-aggregation attention \
    --wandb
```

### Run Multiple Experiments

```bash
python scripts/run_experiments.py --experiments attention_baseline lstm_baseline
```

### Use Hydra Config

```bash
python src/train_hydra.py model=attention training=fast
```

---

## Weights & Biases Integration

W&B is integrated into `src/train.py` for experiment tracking, replacing the need for manual spreadsheets.

### Command Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--wandb` | Enable W&B logging | Disabled |
| `--wandb-project` | Project name in W&B | `deepdiet` |
| `--wandb-run-name` | Custom run name | Auto-generated |
| `--wandb-tags` | Tags for filtering runs | Auto + custom |

### Example Usage

```bash
# Basic W&B logging
python -m src.train --use-side-frames --wandb

# Custom run name and tags
python -m src.train \
    --use-side-frames \
    --side-aggregation attention \
    --wandb \
    --wandb-run-name "attention_lr_sweep_1" \
    --wandb-tags experiment1 attention

# Different project
python -m src.train --use-side-frames --wandb --wandb-project deepdiet-ablations
```

### What Gets Logged

**Automatically tracked:**
- All hyperparameters (model, training, data config)
- Git commit hash
- System info (GPU, Python version)
- Training/validation loss per epoch
- Per-task MAE (calories, mass, fat, carb, protein)
- MAE as percentage of mean
- Timing metrics (epoch time, data loading, forward/backward)
- Best validation loss

**View results at:** `https://wandb.ai/<your-username>/deepdiet`

---

## Experiment Runner

The experiment runner (`scripts/run_experiments.py`) enables batch execution of multiple experiments.

### List Available Experiments

```bash
python scripts/run_experiments.py --list
```

**Pre-defined experiments:**

| Name | Description |
|------|-------------|
| `lstm_baseline` | LSTM aggregation (default) |
| `attention_baseline` | Temporal attention aggregation |
| `mean_baseline` | Mean pooling aggregation |
| `attention_lr_high` | Attention with LR=5e-4 |
| `attention_lr_low` | Attention with LR=5e-5 |
| `attention_frozen` | Attention with frozen encoders |
| `side_only_attention` | Side frames only (ablation) |
| `overhead_only` | Overhead RGB+depth only (ablation) |

### Running Experiments

```bash
# Run all pre-defined experiments
python scripts/run_experiments.py

# Run specific experiments
python scripts/run_experiments.py --experiments attention_baseline lstm_baseline mean_baseline

# Dry run (see commands without executing)
python scripts/run_experiments.py --dry-run

# Custom W&B project
python scripts/run_experiments.py --wandb-project my-project
```

### Output

- Logs saved to `experiment_logs/<experiment>_<timestamp>.log`
- Results summary saved to `experiment_logs/results_<timestamp>.json`
- All runs visible in W&B dashboard

### Adding Custom Experiments

Edit `DEFAULT_EXPERIMENTS` in `scripts/run_experiments.py`:

```python
"my_experiment": {
    "description": "My custom experiment",
    "args": {
        "use_side_frames": True,
        "use_overhead": True,
        "use_depth": True,
        "side_aggregation": "attention",
        "epochs": 30,
        "lr": 2e-4,
    },
    "tags": ["custom", "experiment"],
},
```

Or create a YAML config file:

```yaml
# experiments.yaml
my_experiment:
  description: "My custom experiment"
  args:
    use_side_frames: true
    side_aggregation: attention
    epochs: 30
  tags: ["custom"]
```

```bash
python scripts/run_experiments.py --config experiments.yaml
```

---

## Hydra Configuration

Hydra provides YAML-based configuration with easy overrides and sweeps.

### Config Structure

```
configs/
├── config.yaml          # Main config (composes others)
├── model/
│   ├── default.yaml     # LSTM baseline
│   ├── attention.yaml   # Temporal attention
│   └── side_only.yaml   # Side frames only
├── training/
│   ├── default.yaml     # 20 epochs, batch=4
│   ├── fast.yaml        # 3 epochs (debugging)
│   └── long.yaml        # 50 epochs
├── data/
│   ├── default.yaml     # 256px, 16 frames
│   └── small.yaml       # 128px, 8 frames (faster)
└── logging/
    ├── default.yaml     # W&B enabled
    └── local.yaml       # TensorBoard only
```

### Basic Usage

```bash
# Default configuration
python src/train_hydra.py

# Use attention model config
python src/train_hydra.py model=attention

# Combine configs
python src/train_hydra.py model=attention training=fast data=small

# Quick debug run (no W&B)
python src/train_hydra.py training=fast logging=local
```

### Override Specific Values

```bash
# Override learning rate
python src/train_hydra.py training.lr=5e-5

# Multiple overrides
python src/train_hydra.py model.side_aggregation=attention training.epochs=30 training.lr=2e-4

# Override nested values
python src/train_hydra.py model.freeze_encoders=true model.unfreeze_epoch=5
```

### Hyperparameter Sweeps

```bash
# Sweep over aggregation methods
python src/train_hydra.py --multirun model.side_aggregation=lstm,attention,mean

# Sweep over learning rates
python src/train_hydra.py --multirun training.lr=1e-4,5e-5,1e-5

# Grid search (runs all combinations)
python src/train_hydra.py --multirun \
    model.side_aggregation=lstm,attention \
    training.lr=1e-4,5e-5
```

### Config Options Reference

**Model (`configs/model/`)**
```yaml
use_side_frames: true      # Use side angle frames
use_overhead: true         # Use overhead RGB
use_depth: true            # Use depth images
side_aggregation: lstm     # lstm, attention, or mean
lstm_hidden: 640           # LSTM hidden size
freeze_encoders: false     # Freeze EfficientNet initially
unfreeze_epoch: 10         # When to unfreeze
encoder_lr_multiplier: 0.1 # LR multiplier for encoders
chunk_size: 4              # Frames per forward pass
```

**Training (`configs/training/`)**
```yaml
epochs: 20
batch_size: 4
lr: 1e-4
weight_decay: 1e-4
lr_decay_epochs: 5         # Decay LR every N epochs
lr_decay_factor: 0.5       # Multiply LR by this
max_grad_norm: 5.0         # Gradient clipping
num_workers: 2             # Data loader workers
```

**Data (`configs/data/`)**
```yaml
image_size: 256            # Input image size
max_frames: 16             # Max side frames per dish
data_root: data/nutrition5k_dataset
train_csv: indexes/train_official.csv
test_csv: indexes/test_official.csv
```

**Logging (`configs/logging/`)**
```yaml
wandb: true                # Enable W&B
wandb_project: deepdiet    # W&B project name
wandb_entity: null         # W&B team (null = personal)
tensorboard: true          # Always enabled
```

---

## GCP Workflow

### SSH into Instance

```bash
gcloud compute ssh --zone "us-central1-c" "instance-20251130-082004" --project "cs230-project-478801"
```

### Typical Training Session

```bash
# SSH in
gcloud compute ssh --zone "us-central1-c" "instance-20251130-082004" --project "cs230-project-478801"

# Navigate and update
cd ~/deepdiet
git pull

# Install any new dependencies
pip install -r requirements.txt

# Run training with W&B
python -m src.train \
    --use-side-frames \
    --use-overhead \
    --use-depth \
    --side-aggregation attention \
    --epochs 20 \
    --wandb

# Or run in background
nohup python -m src.train \
    --use-side-frames \
    --side-aggregation attention \
    --wandb \
    > training.log 2>&1 &

# Monitor
tail -f training.log
```

### Running Batch Experiments on GCP

```bash
# SSH in
gcloud compute ssh ...

cd ~/deepdiet && git pull

# Run all experiments (will take hours)
nohup python scripts/run_experiments.py > experiments.log 2>&1 &

# Monitor progress
tail -f experiments.log

# Or check W&B dashboard for real-time updates
```

### Using tmux for Long Runs

```bash
# Start tmux session
tmux new -s training

# Run training
python -m src.train --use-side-frames --wandb

# Detach: Ctrl+B, then D

# Reattach later
tmux attach -t training
```

---

## Comparing Results

### In W&B

1. Go to `https://wandb.ai/<username>/deepdiet`
2. Select runs to compare
3. Use the comparison view to see metrics side-by-side
4. Filter by tags (e.g., "attention", "baseline")

### In TensorBoard

```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser.

---

## Troubleshooting

### W&B Not Logging

```bash
# Check if logged in
wandb login

# Verify API key
cat ~/.netrc | grep wandb
```

### CUDA Out of Memory

```bash
# Reduce batch size
python -m src.train --batch-size 2 --wandb

# Or reduce chunk size (frames processed at once)
python -m src.train --chunk-size 2 --wandb

# Or use smaller images
python src/train_hydra.py data=small
```

### Hydra Working Directory Issues

Hydra changes the working directory. If you have path issues:

```python
# In your code
import hydra
orig_cwd = hydra.utils.get_original_cwd()
```