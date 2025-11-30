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
from src.dataset import MultiViewDataset, TARGETS
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
from src.metrics import (
    compute_per_layer_step_sizes,
    compute_activation_stats,
    compute_distance_from_init,
    compute_gradient_noise,
    compute_mae,
    compute_relative_mae
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
    parser.add_argument('--lstm-hidden', type=int, default=384,
                        help='LSTM hidden size (default: 384)')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for training (default: 256, use 128 or 192 for faster training)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., indexes/side_angles_best.pt)')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='Path to training CSV file (default: use built-in paths based on mode)')
    parser.add_argument('--test-csv', type=str, default=None,
                        help='Path to test CSV file (default: use built-in paths based on mode)')
    args = parser.parse_args()

    config = create_config(args, REPO)

    device = get_device()
    print(f"Device: {device}")

    # Create TensorBoard logger
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    enabled_inputs = config.get_input_channels()
    run_name = f"{'_'.join(enabled_inputs)}_{timestamp}"
    log_dir = REPO / "runs" / run_name
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir runs/")

    if not enabled_inputs:
        print("ERROR: At least one input type must be enabled!")
        return

    print(f"Mode: {' + '.join(enabled_inputs)}")

    # Determine CSV paths (use official splits by default)
    train_csv = config.train_csv
    test_csv = config.test_csv

    # Create datasets
    print("\nLoading datasets...")
    train_ds = MultiViewDataset(
        split_file=train_csv,
        data_root=config.data_root,
        train=True,
        max_side_frames=config.max_frames,
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        image_size=config.image_size
    )
    val_ds = MultiViewDataset(
        split_file=test_csv,
        data_root=config.data_root,
        train=False,
        max_side_frames=config.max_frames,
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        image_size=config.image_size
    )

    # Use 2 workers for data loading
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                         num_workers=config.num_workers, pin_memory=(device.type == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                       num_workers=config.num_workers, pin_memory=(device.type == 'cuda'))

    # Create model
    print("\nInitializing model...")
    model = DeepDietModel(
        use_side_frames=config.use_side_frames,
        use_overhead=config.use_overhead,
        use_depth=config.use_depth,
        chunk_size=config.chunk_size,
        lstm_hidden=config.lstm_hidden,
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

    # Multi-task MAE loss (following Nutrition5k paper)
    criterion = nn.L1Loss(reduction='none')  # Per-element MAE
    optimizer = optim.Adam(
        model.parameters(),
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
            train_mape = train_result['mape']
            train_total_loss = train_result['total_loss']
            data_load_time = train_result['timing']['data_load_time']
            forward_time = train_result['timing']['forward_time']
            backward_time = train_result['timing']['backward_time']
            grad_metrics_accum = train_result['grad_metrics']
            prev_gradients = train_result['prev_gradients']

            # Validate
            model.eval()
            val_losses = {task: 0.0 for task in TARGETS}
            val_mape = {task: 0.0 for task in TARGETS}
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

                    # Track losses and MAPE
                    batch_size = targets.size(0)
                    val_total_loss += weighted_loss.item() * batch_size

                    # Compute MAPE (using compute_relative_mae which returns percentage)
                    relative_mae = compute_relative_mae(pred, targets, per_task=True)

                    for i, task in enumerate(TARGETS):
                        val_losses[task] += task_losses[i].item() * batch_size
                        # convert relative MAE to percentage
                        val_mape[task] += relative_mae[f'task_{i}'] * 100 * batch_size

            # Average losses and MAPE
            val_total_loss /= len(val_ds)
            for task in TARGETS:
                val_losses[task] /= len(val_ds)
                val_mape[task] /= len(val_ds)

            epoch_time = time.time() - epoch_start

            # Average gradient metrics
            num_grad_samples = max(1, batch_count // 10)
            for k in grad_metrics_accum:
                grad_metrics_accum[k] /= num_grad_samples

            # Track gradient norm history for EWMA
            if 'grad_norm' in grad_metrics_accum:
                grad_norm_history.append(grad_metrics_accum['grad_norm'])

            # Print epoch summary
            print(f"\n[Epoch {epoch:02d}/{config.epochs}] Time: {epoch_time:.1f}s (data: {data_load_time:.1f}s, fwd: {forward_time:.1f}s, bwd: {backward_time:.1f}s)")
            print(f"  Total Loss: {train_total_loss:.3f} (train) / {val_total_loss:.3f} (val)")
            print(f"  {'Task':<8} {'MAE (train/val)':<20} {'MAPE % (train/val)'}")
            print(f"  Cal:     {train_losses['cal']:7.2f} / {val_losses['cal']:7.2f}   {train_mape['cal']:6.1f}% / {val_mape['cal']:6.1f}%")
            print(f"  Mass:    {train_losses['mass']:7.2f} / {val_losses['mass']:7.2f}   {train_mape['mass']:6.1f}% / {val_mape['mass']:6.1f}%")
            print(f"  Fat:     {train_losses['fat']:7.2f} / {val_losses['fat']:7.2f}   {train_mape['fat']:6.1f}% / {val_mape['fat']:6.1f}%")
            print(f"  Carb:    {train_losses['carb']:7.2f} / {val_losses['carb']:7.2f}   {train_mape['carb']:6.1f}% / {val_mape['carb']:6.1f}%")
            print(f"  Protein: {train_losses['protein']:7.2f} / {val_losses['protein']:7.2f}   {train_mape['protein']:6.1f}% / {val_mape['protein']:6.1f}%")

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
                writer.add_scalar(f'MAPE_train/{task}', train_mape[task], epoch)
                writer.add_scalar(f'MAPE_val/{task}', val_mape[task], epoch)

            # Flush TensorBoard writer to disk
            writer.flush()
            print(f"  [DEBUG] Logged {len(TARGETS)*2 + 6} metrics to TensorBoard for epoch {epoch}")

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
        return

    print("=" * 70)
    print(f"Training complete! Best validation MAE: {best_val_loss:.3f}")
    writer.close()


if __name__ == "__main__":
    main()
