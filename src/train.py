#!/usr/bin/env python3
"""
Multi-view training script for DeepDiet.
Supports side angle images only, or side angles + overhead RGB/depth.
"""
import warnings

# Suppress torchvision libjpeg warnings before any imports
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')
warnings.filterwarnings('ignore', message='.*Failed to load image Python extension.*')

from src.training.epoch import train_one_epoch
from src.model import DeepDietModel
from src.dataset import MultiViewDataset, TARGETS, multimodal_collate_fn
from src.config import TrainingConfig, create_config

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import argparse
import time
import numpy as np
import os

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run 'pip install wandb' for experiment tracking.")
from src.metrics import (
    compute_per_layer_step_sizes,
    compute_activation_stats,
    compute_distance_from_init,
    compute_gradient_noise,
    compute_target_means
)

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "nutrition5k_dataset"

# Official splits (match Nutrition5k paper)
TRAIN_CSV = REPO / "indexes" / "train_official.csv"
TEST_CSV = REPO / "indexes" / "test_official.csv"

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description='Train DeepDiet multi-view model')
    parser.add_argument('--use-side-frames', action='store_true',
                        help='Use side angle frames (default: enabled)')
    parser.add_argument('--use-overhead', action='store_true',
                        help='Use overhead RGB images (default: disabled)')
    parser.add_argument('--use-depth', action='store_true',
                        help='Use overhead depth images (default: disabled)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--max-frames', type=int, default=16,
                        help='Maximum number of side angle frames per dish (default: 16)')
    parser.add_argument('--chunk-size', type=int, default=4,
                        help='Number of frames to process at once through encoder (default: 4, lower = less memory)')
    parser.add_argument('--lstm-hidden', type=int, default=640,
                        help='LSTM hidden size (default: 640)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for training (default: 256, use 128 or 192 for faster training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., indexes/side_angles_best.pt)')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='Path to training CSV file (default: use built-in paths based on mode)')
    parser.add_argument('--test-csv', type=str, default=None,
                        help='Path to test CSV file (default: use built-in paths based on mode)')
    parser.add_argument('--freeze-encoders', action='store_true',
                        help='Freeze EfficientNet encoders initially (default: False)')
    parser.add_argument('--unfreeze-epoch', type=int, default=10,
                        help='Epoch to unfreeze encoders (default: 10)')
    parser.add_argument('--encoder-lr-multiplier', type=float, default=0.1,
                        help='LR multiplier for encoders when unfrozen (default: 0.1)')
    parser.add_argument('--side-aggregation', type=str, default='lstm', choices=['lstm', 'attention', 'mean'],
                        help='Side frame aggregation method: lstm, attention, or mean (default: lstm)')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping max norm (default: 1.0, use lower for attention)')
    parser.add_argument('--allow-missing-modalities', action='store_true',
                        help='Allow training with partial modalities (uses learned embeddings for missing inputs)')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging (default: disabled)')
    parser.add_argument('--wandb-project', type=str, default='deepdiet',
                        help='W&B project name (default: deepdiet)')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--wandb-tags', type=str, nargs='*', default=[],
                        help='W&B tags for this run (e.g., --wandb-tags baseline attention)')
    args = parser.parse_args()

    config = create_config(args, REPO)

    device = get_device()
    print(f"Device: {device}")

    # Create TensorBoard logger
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    enabled_inputs = config.get_input_channels()
    run_name = f"{'_'.join(enabled_inputs)}_{args.side_aggregation}_{timestamp}"
    log_dir = REPO / "runs" / run_name
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir runs/")

    # Initialize Weights & Biases if enabled
    use_wandb = args.wandb and WANDB_AVAILABLE
    if args.wandb and not WANDB_AVAILABLE:
        print("Warning: --wandb flag set but wandb not installed. Skipping W&B logging.")

    if use_wandb:
        wandb_run_name = args.wandb_run_name or run_name
        wandb_config = {
            # Model architecture
            "side_aggregation": args.side_aggregation,
            "use_side_frames": args.use_side_frames,
            "use_overhead": args.use_overhead,
            "use_depth": args.use_depth,
            "lstm_hidden": args.lstm_hidden,
            "max_frames": args.max_frames,
            "image_size": args.image_size,
            # Training params
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "grad_clip": args.grad_clip,
            "freeze_encoders": args.freeze_encoders,
            "unfreeze_epoch": args.unfreeze_epoch,
            "encoder_lr_multiplier": args.encoder_lr_multiplier,
            "allow_missing_modalities": args.allow_missing_modalities,
            # Environment
            "device": str(device),
        }

        # Auto-generate tags based on config
        auto_tags = list(enabled_inputs) + [args.side_aggregation]
        if args.freeze_encoders:
            auto_tags.append("frozen_encoders")
        all_tags = auto_tags + args.wandb_tags

        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            tags=all_tags,
            sync_tensorboard=True,  # Also sync TensorBoard logs
        )
        print(f"W&B run: {wandb.run.url}")

    if not enabled_inputs:
        print("ERROR: At least one input type must be enabled!")
        return

    print(f"Mode: {' + '.join(enabled_inputs)}")

    # Determine CSV paths (use official splits by default)
    train_csv = config.train_csv
    test_csv = config.test_csv

    # Create datasets
    print("\nLoading datasets...")
    if config.allow_missing_modalities:
        print("Allow missing modalities: ON (using learned embeddings for missing inputs)")
    train_ds = MultiViewDataset(
        split_file=train_csv,
        data_root=config.data_root,
        train=True,
        max_side_frames=config.max_frames,
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        image_size=config.image_size,
        allow_missing_modalities=config.allow_missing_modalities
    )
    val_ds = MultiViewDataset(
        split_file=test_csv,
        data_root=config.data_root,
        train=False,
        max_side_frames=config.max_frames,
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        image_size=config.image_size,
        allow_missing_modalities=config.allow_missing_modalities
    )

    # Use 2 workers for data loading
    # Use custom collate function when allowing missing modalities
    collate_fn = multimodal_collate_fn if config.allow_missing_modalities else None
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                         num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
                         collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                       num_workers=config.num_workers, pin_memory=(device.type == 'cuda'),
                       collate_fn=collate_fn)

    # Compute mean target values for percentage calculations (matching paper methodology)
    print("\nComputing dataset statistics...")
    target_means = compute_target_means(train_ds, TARGETS)
    print("Training set mean values:")
    for task in TARGETS:
        print(f"  {task}: {target_means[task]:.2f}")

    # Create model
    print("\nInitializing model...")
    print(f"Side aggregation: {config.side_aggregation}")
    model = DeepDietModel(
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        chunk_size=config.chunk_size,
        lstm_hidden=config.lstm_hidden,
        side_aggregation=config.side_aggregation,
        allow_missing_modalities=config.allow_missing_modalities,
    ).to(device)

    # Load checkpoint if resuming
    start_epoch = 1
    if config.resume_from:
        print(f"Loading checkpoint from {config.resume_from}...")
        checkpoint_path = config.resume_from
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Resumed from checkpoint: {config.resume_from}")
        else:
            print(f"Checkpoint not found: {config.resume_from}, starting from scratch")

    # Freeze encoders if configured
    encoders_frozen = False
    if config.freeze_encoders:
        model.freeze_encoders()
        encoders_frozen = True
        frozen_params = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable_params = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"Encoders frozen: {frozen_params} params frozen, {trainable_params} params trainable")
        print(f"Will unfreeze at epoch {config.unfreeze_epoch} with encoder LR multiplier {config.encoder_lr_multiplier}")

    # Multi-task MAE loss (following Nutrition5k paper)
    criterion = nn.L1Loss(reduction='none')  # Per-element MAE

    # Create optimizer (only for trainable params initially)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Use automatic mixed precision to reduce memory usage
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    task_weights = config.task_weights

    # Store initial weights for distance tracking
    print("Saving initial weights...")
    initial_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters() if param.requires_grad}

    # Training loop
    best_val_loss = math.inf

    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Loss: Multi-task MAE (cal, mass, fat, carb, protein)")
    print("=" * 70)

    # Tracking for EWMA gradient metrics
    grad_norm_history = []
    prev_gradients = None
    batch_grad_norms = []  # Track per-batch gradient norms for noise estimation

    try:
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()

            # Unfreeze encoders at the specified epoch
            if encoders_frozen and epoch == config.unfreeze_epoch:
                print(f"\n  *** Unfreezing encoders at epoch {epoch} ***")
                model.unfreeze_encoders()
                encoders_frozen = False

                # Rebuild optimizer with discriminative learning rates
                param_groups = model.get_param_groups(
                    base_lr=config.learning_rate,
                    encoder_lr_multiplier=config.encoder_lr_multiplier
                )
                optimizer = optim.Adam(
                    param_groups,
                    weight_decay=config.weight_decay
                )

                # Apply same LR decay that would have happened so far
                num_decays = (epoch - 1) // config.lr_decay_epochs
                for _ in range(num_decays):
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= config.lr_decay_factor

                encoder_lr = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else 0
                other_lr = optimizer.param_groups[0]['lr']
                print(f"  New optimizer: encoder LR = {encoder_lr:.6f}, other LR = {other_lr:.6f}")

            train_result = train_one_epoch(
                model=model,
                dataloader=train_dl,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                config=config,
                task_weights=task_weights,
                epoch=epoch,
                scaler=scaler,
                prev_gradients=prev_gradients,
                batch_grad_norms=batch_grad_norms
            )

            train_losses = train_result['losses']
            train_total_loss = train_result['total_loss']
            data_load_time = train_result['timing']['data_load_time']
            forward_time = train_result['timing']['forward_time']
            backward_time = train_result['timing']['backward_time']
            grad_metrics_accum = train_result['grad_metrics']
            prev_gradients = train_result['prev_gradients']

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

                    # Use mixed precision for validation too
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            pred = model(side_frames, overhead_rgb, overhead_depth)
                            # Compute per-task MAE loss
                            task_losses = criterion(pred, targets).mean(dim=0)
                            # Weighted sum
                            weighted_loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))
                    else:
                        pred = model(side_frames, overhead_rgb, overhead_depth)
                        # Compute per-task MAE loss
                        task_losses = criterion(pred, targets).mean(dim=0)
                        # Weighted sum
                        weighted_loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))

                    # Track losses
                    batch_size = targets.size(0)
                    val_total_loss += weighted_loss.item() * batch_size

                    for i, task in enumerate(TARGETS):
                        val_losses[task] += task_losses[i].item() * batch_size

            # Average losses
            val_total_loss /= len(val_ds)
            for task in TARGETS:
                val_losses[task] /= len(val_ds)

            epoch_time = time.time() - epoch_start

            # Track gradient norm history for EWMA
            if 'grad_norm' in grad_metrics_accum:
                grad_norm_history.append(grad_metrics_accum['grad_norm'])

            # Print epoch summary
            print(f"\n[Epoch {epoch:02d}/{config.epochs}] Time: {epoch_time:.1f}s (data: {data_load_time:.1f}s, fwd: {forward_time:.1f}s, bwd: {backward_time:.1f}s)")
            print(f"  Total Loss: {train_total_loss:.3f} (train) / {val_total_loss:.3f} (val)")
            print(f"  {'Task':<8} {'MAE (train/val)':<20} {'MAE % of mean (train/val)'}")
            print(f"  Cal:     {train_losses['cal']:7.2f} / {val_losses['cal']:7.2f}   {train_losses['cal']/target_means['cal']*100:6.1f}% / {val_losses['cal']/target_means['cal']*100:6.1f}%")
            print(f"  Mass:    {train_losses['mass']:7.2f} / {val_losses['mass']:7.2f}   {train_losses['mass']/target_means['mass']*100:6.1f}% / {val_losses['mass']/target_means['mass']*100:6.1f}%")
            print(f"  Fat:     {train_losses['fat']:7.2f} / {val_losses['fat']:7.2f}   {train_losses['fat']/target_means['fat']*100:6.1f}% / {val_losses['fat']/target_means['fat']*100:6.1f}%")
            print(f"  Carb:    {train_losses['carb']:7.2f} / {val_losses['carb']:7.2f}   {train_losses['carb']/target_means['carb']*100:6.1f}% / {val_losses['carb']/target_means['carb']*100:6.1f}%")
            print(f"  Protein: {train_losses['protein']:7.2f} / {val_losses['protein']:7.2f}   {train_losses['protein']/target_means['protein']*100:6.1f}% / {val_losses['protein']/target_means['protein']*100:6.1f}%")

            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_total_loss, epoch)
            writer.add_scalar('Loss/val', val_total_loss, epoch)
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            writer.add_scalar('Time/data_load', data_load_time, epoch)
            writer.add_scalar('Time/forward', forward_time, epoch)
            writer.add_scalar('Time/backward', backward_time, epoch)

            for task in TARGETS:
                writer.add_scalar(f'MAE_train/{task}', train_losses[task], epoch)
                writer.add_scalar(f'MAE_val/{task}', val_losses[task], epoch)
                # Log MAE as percentage of mean (matching paper methodology)
                writer.add_scalar(f'MAE_percent_train/{task}', train_losses[task]/target_means[task]*100, epoch)
                writer.add_scalar(f'MAE_percent_val/{task}', val_losses[task]/target_means[task]*100, epoch)

            # Flush TensorBoard writer to disk
            writer.flush()

            # Log to W&B (in addition to TensorBoard sync)
            if use_wandb:
                wandb_metrics = {
                    "epoch": epoch,
                    "train/loss": train_total_loss,
                    "val/loss": val_total_loss,
                    "time/epoch": epoch_time,
                    "time/data_load": data_load_time,
                    "time/forward": forward_time,
                    "time/backward": backward_time,
                    "best_val_loss": best_val_loss,
                }
                for task in TARGETS:
                    wandb_metrics[f"train/mae_{task}"] = train_losses[task]
                    wandb_metrics[f"val/mae_{task}"] = val_losses[task]
                    wandb_metrics[f"train/mae_pct_{task}"] = train_losses[task] / target_means[task] * 100
                    wandb_metrics[f"val/mae_pct_{task}"] = val_losses[task] / target_means[task] * 100
                wandb.log(wandb_metrics, step=epoch)

            print(f"  [DEBUG] Logged {len(TARGETS)*4 + 6} metrics to TensorBoard for epoch {epoch}")

            # Compute additional metrics (do this every few epochs to minimize overhead)
            if epoch % config.advanced_metrics_freq == 0:
                # Gradient noise
                noise_metrics = compute_gradient_noise(batch_grad_norms, window=config.grad_noise_window)
                grad_noise_cv = noise_metrics['grad_noise_cv']

                # Per-layer step sizes
                current_lr = optimizer.param_groups[0]['lr']
                layer_steps = compute_per_layer_step_sizes(model, current_lr, grouped=True)

                # Distance from initialization
                dist_from_init = compute_distance_from_init(model, initial_weights, grouped=True)

                # Activation stats (use first validation batch to avoid modifying training flow)
                try:
                    first_val_batch = next(iter(val_dl))
                    val_side_frames = first_val_batch['side_frames'][:2].to(device)  # Just 2 samples
                    val_overhead_rgb = first_val_batch.get('overhead_rgb')
                    val_overhead_depth = first_val_batch.get('overhead_depth')
                    if val_overhead_rgb is not None:
                        val_overhead_rgb = val_overhead_rgb[:2].to(device)
                    if val_overhead_depth is not None:
                        val_overhead_depth = val_overhead_depth[:2].to(device)

                    model.eval()
                    activation_result = compute_activation_stats(model, val_side_frames, val_overhead_rgb, val_overhead_depth, sample_rate=config.activation_sample_rate)
                    model.train()

                    # Get average dead fraction
                    avg_dead_fraction = activation_result.get('avg_dead_fraction', 0.0)
                except:
                    avg_dead_fraction = 0
                    layer_steps = {}
                    dist_from_init = {}

            # Print gradient metrics
            if grad_metrics_accum:
                print(f"  [Gradients] Norm: {grad_metrics_accum.get('grad_norm', 0):.4f}  |  " +
                      f"L2 (RMS): {grad_metrics_accum.get('grad_l2', 0):.6f}  |  " +
                      f"Effective Dim: {grad_metrics_accum.get('effective_dim', 0):.2f}")

                # Log gradient metrics to TensorBoard
                writer.add_scalar('Gradients/norm', grad_metrics_accum.get('grad_norm', 0), epoch)
                writer.add_scalar('Gradients/l2', grad_metrics_accum.get('grad_l2', 0), epoch)
                writer.add_scalar('Gradients/l1_l2_ratio', grad_metrics_accum.get('grad_l1_l2_ratio', 0), epoch)
                writer.add_scalar('Gradients/effective_dim', grad_metrics_accum.get('effective_dim', 0), epoch)
                writer.add_scalar('Gradients/mean', grad_metrics_accum.get('grad_mean', 0), epoch)
                writer.add_scalar('Gradients/std', grad_metrics_accum.get('grad_std', 0), epoch)

                if 'grad_cosine_sim' in grad_metrics_accum:
                    print(f"  [Gradients] Cosine Similarity: {grad_metrics_accum['grad_cosine_sim']:.4f}  " +
                          f"(1.0 = consistent direction, -1.0 = flipping)")
                    writer.add_scalar('Gradients/cosine_sim', grad_metrics_accum['grad_cosine_sim'], epoch)

                # Compute EWMA and EWSD if we have history
                if len(grad_norm_history) >= 3:
                    from src.metrics import update_ewma
                    ewma_metrics = update_ewma(grad_norm_history, window=10)
                    print(f"  [Gradients] EWMA: {ewma_metrics['ewma']:.4f}  |  EWSD: {ewma_metrics['ewsd']:.4f}")
                    writer.add_scalar('Gradients/ewma', ewma_metrics['ewma'], epoch)
                    writer.add_scalar('Gradients/ewsd', ewma_metrics['ewsd'], epoch)

            # Print additional metrics every 2 epochs
            if epoch % 2 == 0:
                print(f"  [Gradients] Noise CV: {grad_noise_cv:.4f}  |  [Activations] Dead ReLUs: {avg_dead_fraction:.2%}")

                # Log additional metrics to TensorBoard
                writer.add_scalar('Gradients/noise_cv', grad_noise_cv, epoch)
                writer.add_scalar('Activations/dead_relu_fraction', avg_dead_fraction, epoch)

                # Per-layer step sizes
                if layer_steps:
                    print(f"  [Layer Step Sizes] ", end="")
                    for layer, stats in layer_steps.items():
                        print(f"{layer}={stats['mean_step']:.6f} ", end="")
                        writer.add_scalar(f'LayerSteps/{layer}_mean', stats['mean_step'], epoch)
                        writer.add_scalar(f'LayerSteps/{layer}_max', stats['max_step'], epoch)
                    print()

                # Distance from init
                if dist_from_init:
                    print(f"  [Distance from Init] ", end="")
                    for layer, stats in dist_from_init.items():
                        print(f"{layer}={stats['mean_dist']:.3f} ", end="")
                        writer.add_scalar(f'Distance/{layer}_mean', stats['mean_dist'], epoch)
                        writer.add_scalar(f'Distance/{layer}_max', stats['max_dist'], epoch)
                    print()

            # Clear batch grad norms after each epoch
            batch_grad_norms = []

            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                model_name = "multiview_best.pt"
                save_path = REPO / "indexes" / model_name
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model to {save_path}")

            # Learning rate decay
            if epoch % config.lr_decay_epochs == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= config.lr_decay_factor
                    print(f"  → Learning rate: {param_group['lr']:.6f}")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted by user (Ctrl+C)")
        if best_val_loss < math.inf:
            print(f"Best validation MAE so far: {best_val_loss:.3f}")
            model_name = "multiview_best.pt"
            print(f"Best model saved at: {REPO / 'indexes' / model_name}")
        print("=" * 70)
        writer.close()
        if use_wandb:
            wandb.finish(exit_code=1)
        return

    print("=" * 70)
    print(f"Training complete! Best validation MAE: {best_val_loss:.3f}")
    writer.close()

    # Finish W&B run and log final summary
    if use_wandb:
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["final_train_loss"] = train_total_loss
        for task in TARGETS:
            wandb.summary[f"best_val_mae_{task}"] = val_losses[task]
        wandb.finish()


if __name__ == "__main__":
    main()
