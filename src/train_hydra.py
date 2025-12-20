#!/usr/bin/env python3
"""
Hydra-based training script for DeepDiet.

This script uses Hydra for configuration management, enabling:
- YAML-based config files with overrides
- Automatic experiment tracking with W&B
- Easy hyperparameter sweeps

Usage:
    # Default training
    python src/train_hydra.py

    # Use attention model
    python src/train_hydra.py model=attention

    # Override specific values
    python src/train_hydra.py model.side_aggregation=attention training.lr=5e-5

    # Fast debug run
    python src/train_hydra.py training=fast data=small logging=local

    # Multirun sweep
    python src/train_hydra.py --multirun model.side_aggregation=lstm,attention,mean training.lr=1e-4,5e-5
"""
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')
warnings.filterwarnings('ignore', message='.*Failed to load image Python extension.*')

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import time
import os

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.training.epoch import train_one_epoch
from src.model import DeepDietModel
from src.dataset import MultiViewDataset, TARGETS, multimodal_collate_fn
from src.metrics import compute_target_means


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print resolved config
    print(OmegaConf.to_yaml(cfg))

    # Get original working directory (Hydra changes cwd)
    orig_cwd = hydra.utils.get_original_cwd()
    repo = Path(orig_cwd)

    device = get_device()
    print(f"Device: {device}")

    # Generate run name
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    modalities = []
    if cfg.model.use_side_frames:
        modalities.append("side")
    if cfg.model.use_overhead:
        modalities.append("overhead")
    if cfg.model.use_depth:
        modalities.append("depth")

    run_name = cfg.experiment.name or f"{'_'.join(modalities)}_{cfg.model.side_aggregation}_{timestamp}"

    # TensorBoard setup
    log_dir = repo / "runs" / run_name
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")

    # W&B setup
    use_wandb = cfg.logging.wandb and WANDB_AVAILABLE
    if cfg.logging.wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Skipping W&B logging.")

    if use_wandb:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        tags = modalities + [cfg.model.side_aggregation] + list(cfg.experiment.tags)

        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=run_name,
            config=wandb_config,
            tags=tags,
            notes=cfg.experiment.notes,
            sync_tensorboard=True,
        )
        print(f"W&B run: {wandb.run.url}")

    # Data paths
    data_root = repo / cfg.data.data_root
    train_csv = repo / cfg.data.train_csv
    test_csv = repo / cfg.data.test_csv

    # Datasets
    print("\nLoading datasets...")
    train_ds = MultiViewDataset(
        split_file=train_csv,
        data_root=data_root,
        train=True,
        max_side_frames=cfg.data.max_frames,
        use_side_frames=cfg.model.use_side_frames,
        use_overhead=cfg.model.use_overhead,
        use_depth=cfg.model.use_depth,
        image_size=cfg.data.image_size,
        allow_missing_modalities=cfg.model.allow_missing_modalities,
    )
    val_ds = MultiViewDataset(
        split_file=test_csv,
        data_root=data_root,
        train=False,
        max_side_frames=cfg.data.max_frames,
        use_side_frames=cfg.model.use_side_frames,
        use_overhead=cfg.model.use_overhead,
        use_depth=cfg.model.use_depth,
        image_size=cfg.data.image_size,
        allow_missing_modalities=cfg.model.allow_missing_modalities,
    )

    collate_fn = multimodal_collate_fn if cfg.model.allow_missing_modalities else None
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn,
    )

    # Target means for percentage metrics
    print("\nComputing dataset statistics...")
    target_means = compute_target_means(train_ds, TARGETS)
    for task in TARGETS:
        print(f"  {task}: {target_means[task]:.2f}")

    # Model
    print(f"\nInitializing model with {cfg.model.side_aggregation} aggregation...")
    model = DeepDietModel(
        use_side_frames=cfg.model.use_side_frames,
        use_overhead=cfg.model.use_overhead,
        use_depth=cfg.model.use_depth,
        chunk_size=cfg.model.chunk_size,
        lstm_hidden=cfg.model.lstm_hidden,
        side_aggregation=cfg.model.side_aggregation,
        allow_missing_modalities=cfg.model.allow_missing_modalities,
    ).to(device)

    # Freeze encoders if configured
    encoders_frozen = False
    if cfg.model.freeze_encoders:
        model.freeze_encoders()
        encoders_frozen = True
        print(f"Encoders frozen until epoch {cfg.model.unfreeze_epoch}")

    # Optimizer and criterion
    criterion = nn.L1Loss(reduction='none')
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    task_weights = {task: 1.0 for task in TARGETS}

    # Training loop
    best_val_loss = math.inf
    print(f"\nStarting training for {cfg.training.epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start = time.time()

        # Unfreeze encoders
        if encoders_frozen and epoch == cfg.model.unfreeze_epoch:
            print(f"\n  *** Unfreezing encoders at epoch {epoch} ***")
            model.unfreeze_encoders()
            encoders_frozen = False
            param_groups = model.get_param_groups(
                base_lr=cfg.training.lr,
                encoder_lr_multiplier=cfg.model.encoder_lr_multiplier,
            )
            optimizer = optim.Adam(param_groups, weight_decay=cfg.training.weight_decay)

        # Train
        model.train()
        train_losses = {task: 0.0 for task in TARGETS}
        train_total_loss = 0.0

        for batch in train_dl:
            targets = batch['targets'].to(device)
            side_frames = batch.get('side_frames')
            overhead_rgb = batch.get('overhead_rgb')
            overhead_depth = batch.get('overhead_depth')

            if side_frames is not None:
                side_frames = side_frames.to(device)
            if overhead_rgb is not None:
                overhead_rgb = overhead_rgb.to(device)
            if overhead_depth is not None:
                overhead_depth = overhead_depth.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    pred = model(side_frames, overhead_rgb, overhead_depth)
                    task_losses = criterion(pred, targets).mean(dim=0)
                    loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(side_frames, overhead_rgb, overhead_depth)
                task_losses = criterion(pred, targets).mean(dim=0)
                loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()

            batch_size = targets.size(0)
            train_total_loss += loss.item() * batch_size
            for i, task in enumerate(TARGETS):
                train_losses[task] += task_losses[i].item() * batch_size

        train_total_loss /= len(train_ds)
        for task in TARGETS:
            train_losses[task] /= len(train_ds)

        # Validate
        model.eval()
        val_losses = {task: 0.0 for task in TARGETS}
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in val_dl:
                targets = batch['targets'].to(device)
                side_frames = batch.get('side_frames')
                overhead_rgb = batch.get('overhead_rgb')
                overhead_depth = batch.get('overhead_depth')

                if side_frames is not None:
                    side_frames = side_frames.to(device)
                if overhead_rgb is not None:
                    overhead_rgb = overhead_rgb.to(device)
                if overhead_depth is not None:
                    overhead_depth = overhead_depth.to(device)

                if scaler:
                    with torch.cuda.amp.autocast():
                        pred = model(side_frames, overhead_rgb, overhead_depth)
                        task_losses = criterion(pred, targets).mean(dim=0)
                        loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))
                else:
                    pred = model(side_frames, overhead_rgb, overhead_depth)
                    task_losses = criterion(pred, targets).mean(dim=0)
                    loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))

                batch_size = targets.size(0)
                val_total_loss += loss.item() * batch_size
                for i, task in enumerate(TARGETS):
                    val_losses[task] += task_losses[i].item() * batch_size

        val_total_loss /= len(val_ds)
        for task in TARGETS:
            val_losses[task] /= len(val_ds)

        epoch_time = time.time() - epoch_start

        # Print summary
        print(f"\n[Epoch {epoch:02d}/{cfg.training.epochs}] Time: {epoch_time:.1f}s")
        print(f"  Loss: {train_total_loss:.3f} (train) / {val_total_loss:.3f} (val)")

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_total_loss, epoch)
        writer.add_scalar('Loss/val', val_total_loss, epoch)
        for task in TARGETS:
            writer.add_scalar(f'MAE_train/{task}', train_losses[task], epoch)
            writer.add_scalar(f'MAE_val/{task}', val_losses[task], epoch)
        writer.flush()

        # Log to W&B
        if use_wandb:
            wandb_metrics = {
                "epoch": epoch,
                "train/loss": train_total_loss,
                "val/loss": val_total_loss,
            }
            for task in TARGETS:
                wandb_metrics[f"train/mae_{task}"] = train_losses[task]
                wandb_metrics[f"val/mae_{task}"] = val_losses[task]
            wandb.log(wandb_metrics, step=epoch)

        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_path = repo / "indexes" / "multiview_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_loss: {best_val_loss:.3f})")

        # LR decay
        if epoch % cfg.training.lr_decay_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.training.lr_decay_factor

    print("=" * 70)
    print(f"Training complete! Best val loss: {best_val_loss:.3f}")
    writer.close()

    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()

    return best_val_loss


if __name__ == "__main__":
    main()