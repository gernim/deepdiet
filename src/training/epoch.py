import time
from collections import defaultdict
from typing import Dict

import torch
from tqdm import tqdm
from src.dataset import MultiViewDataset, TARGETS
from src.metrics import compute_gradient_metrics, compute_relative_mae


def train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer,
        device,
        config,
        task_weights,
        epoch,
        scaler,
        prev_gradients=None,
        batch_grad_norms=None
):
    model.train()
    train_losses = {task: 0.0 for task in TARGETS}
    train_mape = {task: 0.0 for task in TARGETS}
    train_total_loss = 0.0

    data_load_time, forward_time, backward_time = 0.0, 0.0, 0.0
    batch_count = 0
    grad_metrics_accum = defaultdict(float)

    batch_start = time.time()
    train_pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)

    for batch in train_pbar:
        batch: Dict[str, torch.Tensor]
        data_load_time += time.time() - batch_start
        batch_count += 1
        forward_start = time.time()

        targets = batch['targets'].to(device)
        side_frames = batch.get('side_frames')
        side_frames = side_frames.to(device) if side_frames is not None else None
        overhead_rgb = batch.get('overhead_rgb')
        overhead_rgb = overhead_rgb.to(device) if overhead_rgb is not None else None
        overhead_depth = batch.get('overhead_depth')
        overhead_depth = overhead_depth.to(device) if overhead_depth is not None else None

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(side_frames, overhead_rgb, overhead_depth)

        forward_time += time.time() - forward_start
        task_losses = criterion(pred, targets).mean(dim=0)
        weighted_losses = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))

        backward_start = time.time()
        scaler.scale(weighted_losses).backward()
        scaler.unscale_(optimizer)

        if batch_count % config.grad_metrics_freq == 0:
            grad_metrics, new_grads = compute_gradient_metrics(model, prev_gradients)
            for k, v in grad_metrics.items():
                grad_metrics_accum[k] += v
            prev_gradients = new_grads

        batch_grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
        if batch_grad_norms is not None:
            batch_grad_norms.append(batch_grad_norm)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        backward_time += time.time() - backward_start

        batch_size = targets.size(0)
        train_total_loss += weighted_losses.item() * batch_size
        relative_mae = compute_relative_mae(pred, targets, per_task=True)
        for i, task in enumerate(TARGETS):
            train_losses[task] += task_losses[i].item() * batch_size
            train_mape[task] += relative_mae[f'task_{i}'] * 100 * batch_size

        train_pbar.set_postfix({
            'loss': f'{weighted_losses.item():.2f}',
            'data': f'{data_load_time / batch_count:.2f}s',
            'fwd': f'{forward_time / batch_count:.2f}s',
            'bwd': f'{backward_time / batch_count:.2f}s'
        })

        batch_start = time.time()

    # Average losses and MAPE
    dataset_size = len(dataloader.dataset)
    train_total_loss /= dataset_size
    for task in TARGETS:
        train_losses[task] /= dataset_size
        train_mape[task] /= dataset_size

    # Average gradient metrics
    num_grad_samples = max(1, batch_count // config.grad_metrics_freq)
    for k in grad_metrics_accum:
        grad_metrics_accum[k] /= num_grad_samples

    return {
        'losses': train_losses,
        'mape': train_mape,
        'total_loss': train_total_loss,
        'timing': {
            'data_load_time': data_load_time,
            'forward_time': forward_time,
            'backward_time': backward_time
        },
        'grad_metrics': grad_metrics_accum,
        'prev_gradients': prev_gradients
    }