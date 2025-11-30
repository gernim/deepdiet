"""
DeepDiet Metrics Module

All metric computations for training monitoring:
- Task Performance: MAE, relative MAE, median AE, signed error
- Gradient Metrics: Norms, EWMA, cosine sim, noise, CV
- Layer Metrics: Weight norms, step sizes, distance from init
- Activation Metrics: Dead ReLUs, activation stats
- Timing Metrics: Timing tracker and throughput
"""
import torch
import torch.nn as nn
import math
import numpy as np
import time
from collections import deque


# ============================================================================
# A. TASK PERFORMANCE METRICS
# ============================================================================

def compute_mae(predictions, targets, per_task=True):
    """
    Mean Absolute Error

    Equation: MAE = (1/N) * sum(|y_i - ŷ_i|)

    Args:
        predictions: (batch_size, num_tasks) tensor
        targets: (batch_size, num_tasks) tensor
        per_task: If True, return MAE per task; else return global MAE

    Returns:
        dict: {task_idx: mae_value} if per_task, else {'mae': global_value}
    """
    errors = (predictions - targets).abs()

    if per_task:
        mae_per_task = errors.mean(dim=0)
        return {f'task_{i}': mae_per_task[i].item() for i in range(len(mae_per_task))}
    else:
        return {'mae': errors.mean().item()}


def compute_target_means(dataset, task_names):
    """
    Compute mean target values across entire dataset.
    Used for computing MAE as percentage of mean (as in Nutrition5k paper).

    Args:
        dataset: Dataset object (not DataLoader) - accesses .df directly to avoid loading images
        task_names: List of task names (e.g., ['cal', 'mass', 'fat', 'carb', 'protein'])

    Returns:
        dict: {task_name: mean_value}
    """
    # Access the dataframe directly to avoid loading images
    df = dataset.df
    target_means = {}

    for task in task_names:
        target_means[task] = df[task].mean()

    return target_means


def compute_relative_mae(predictions, targets, epsilon=1e-6, per_task=True):
    """
    Relative Mean Absolute Error (scale-free)

    Equation: Relative MAE = |y - ŷ| / (|y| + ε)

    Args:
        predictions: (batch_size, num_tasks) tensor
        targets: (batch_size, num_tasks) tensor
        epsilon: Small constant to avoid division by zero
        per_task: If True, return per-task metrics

    Returns:
        dict: {task_idx: relative_mae} if per_task
    """
    relative_errors = (predictions - targets).abs() / (targets.abs() + epsilon)

    if per_task:
        rel_mae_per_task = relative_errors.mean(dim=0)
        return {f'task_{i}': rel_mae_per_task[i].item() for i in range(len(rel_mae_per_task))}
    else:
        return {'relative_mae': relative_errors.mean().item()}


def compute_median_ae(predictions, targets, per_task=True):
    """
    Median Absolute Error (robust to outliers)

    Equation: Median(|y - ŷ|)

    Args:
        predictions: (batch_size, num_tasks) tensor
        targets: (batch_size, num_tasks) tensor
        per_task: If True, return per-task metrics

    Returns:
        dict: {task_idx: median_ae}
    """
    errors = (predictions - targets).abs()

    if per_task:
        median_per_task = errors.median(dim=0)[0]
        return {f'task_{i}': median_per_task[i].item() for i in range(len(median_per_task))}
    else:
        return {'median_ae': errors.median().item()}


def compute_mean_signed_error(predictions, targets, per_task=True):
    """
    Mean Signed Error (reveals systematic bias)

    Equation: Mean(ŷ - y)

    Positive = overestimation, Negative = underestimation

    Args:
        predictions: (batch_size, num_tasks) tensor
        targets: (batch_size, num_tasks) tensor
        per_task: If True, return per-task metrics

    Returns:
        dict: {task_idx: mean_signed_error}
    """
    signed_errors = predictions - targets

    if per_task:
        mse_per_task = signed_errors.mean(dim=0)
        return {f'task_{i}': mse_per_task[i].item() for i in range(len(mse_per_task))}
    else:
        return {'mean_signed_error': signed_errors.mean().item()}


def compute_task_loss_balance(task_losses, task_weights=None):
    """
    Task Loss Balance (multi-task diagnostic)

    Equation: Task share = L_t / sum(L_k)

    Shows if one task dominates training.

    Args:
        task_losses: dict or list of losses per task
        task_weights: Optional dict/list of task weights

    Returns:
        dict: {task_idx: share_of_total_loss}
    """
    if isinstance(task_losses, dict):
        losses = list(task_losses.values())
        task_names = list(task_losses.keys())
    else:
        losses = task_losses
        task_names = [f'task_{i}' for i in range(len(losses))]

    # Apply weights if provided
    if task_weights is not None:
        if isinstance(task_weights, dict):
            weights = [task_weights.get(name, 1.0) for name in task_names]
        else:
            weights = task_weights
        weighted_losses = [l * w for l, w in zip(losses, weights)]
    else:
        weighted_losses = losses

    total_loss = sum(weighted_losses)

    if total_loss == 0:
        return {name: 0.0 for name in task_names}

    return {name: (wl / total_loss) for name, wl in zip(task_names, weighted_losses)}


class TaskMetricsAccumulator:
    """
    Helper class to accumulate metrics over multiple batches.

    Usage:
        acc = TaskMetricsAccumulator(task_names=['cal', 'mass', 'fat', 'carb', 'protein'])

        for batch in dataloader:
            predictions, targets = model(batch), batch['targets']
            acc.update(predictions, targets, batch_size=len(predictions))

        metrics = acc.compute()
    """

    def __init__(self, task_names=None):
        self.task_names = task_names
        self.reset()

    def reset(self):
        """Clear accumulated values."""
        self.total_mae = []
        self.total_relative_mae = []
        self.total_signed_errors = []
        self.all_errors = []
        self.total_samples = 0

    def update(self, predictions, targets, batch_size=None):
        """
        Add batch predictions and targets.

        Args:
            predictions: (batch_size, num_tasks) tensor
            targets: (batch_size, num_tasks) tensor
            batch_size: Optional batch size (default: inferred from predictions)
        """
        if batch_size is None:
            batch_size = predictions.size(0)

        with torch.no_grad():
            errors = (predictions - targets).abs()
            self.all_errors.append(errors.cpu())

            mae = errors.sum(dim=0)
            self.total_mae.append(mae.cpu())

            rel_errors = errors / (targets.abs() + 1e-6)
            self.total_relative_mae.append(rel_errors.sum(dim=0).cpu())

            signed = (predictions - targets).sum(dim=0)
            self.total_signed_errors.append(signed.cpu())

            self.total_samples += batch_size

    def compute(self):
        """
        Compute averaged metrics across all accumulated batches.

        Returns:
            dict: Contains 'mae', 'relative_mae', 'median_ae', 'mean_signed_error' per task
        """
        if self.total_samples == 0:
            return {}

        num_tasks = len(self.total_mae[0])

        total_mae_sum = torch.stack(self.total_mae).sum(dim=0)
        mae_per_task = total_mae_sum / self.total_samples

        total_rel_mae_sum = torch.stack(self.total_relative_mae).sum(dim=0)
        rel_mae_per_task = total_rel_mae_sum / self.total_samples

        total_signed_sum = torch.stack(self.total_signed_errors).sum(dim=0)
        signed_per_task = total_signed_sum / self.total_samples

        all_errors_cat = torch.cat(self.all_errors, dim=0)
        median_per_task = all_errors_cat.median(dim=0)[0]

        result = {}
        for i in range(num_tasks):
            task_name = self.task_names[i] if self.task_names else f'task_{i}'
            result[f'{task_name}_mae'] = mae_per_task[i].item()
            result[f'{task_name}_relative_mae'] = rel_mae_per_task[i].item()
            result[f'{task_name}_median_ae'] = median_per_task[i].item()
            result[f'{task_name}_signed_error'] = signed_per_task[i].item()

        return result


# ============================================================================
# B. GRADIENT & OPTIMIZATION METRICS
# ============================================================================

def compute_gradient_metrics(model, prev_gradients=None):
    """
    Compute comprehensive gradient statistics.

    Metrics computed:
    - grad_norm: Global L2 norm of all gradients
    - grad_l2: RMS of gradient entries
    - grad_l1_l2_ratio: Sparsity measure
    - effective_dim: Gradient alignment measure
    - grad_mean: Average gradient value
    - grad_std: Standard deviation of gradients
    - grad_cosine_sim: Cosine similarity with previous gradient (if provided)

    Args:
        model: PyTorch model with computed gradients
        prev_gradients: Optional list of previous gradient tensors for cosine similarity

    Returns:
        metrics: dict of gradient metrics
        current_gradients: list of flattened gradient tensors (for next iteration)
    """
    total_norm = 0.0
    total_grad_sum = 0.0
    total_grad_sq = 0.0
    total_abs_grad = 0.0
    num_params = 0

    grad_list = []
    current_gradients = []

    for p in model.parameters():
        if p.grad is not None:
            grad = p.grad.data
            grad_flat = grad.flatten()
            grad_list.append(grad_flat)
            current_gradients.append(grad_flat.clone())

            total_norm += grad.norm(2).item() ** 2
            total_grad_sum += grad.sum().item()
            total_grad_sq += (grad ** 2).sum().item()
            total_abs_grad += grad.abs().sum().item()
            num_params += grad.numel()

    if num_params == 0:
        return {}, None

    all_grads = torch.cat(grad_list)
    current_grads_tensor = torch.cat(current_gradients)

    grad_norm = math.sqrt(total_norm)
    l2_metric = math.sqrt(total_grad_sq / num_params)
    l1_l2_ratio = total_abs_grad / (l2_metric * math.sqrt(num_params)) if l2_metric > 0 else 0
    effective_dim = (total_grad_sum ** 2) / total_grad_sq if total_grad_sq > 0 else 0

    metrics = {
        'grad_norm': grad_norm,
        'grad_l2': l2_metric,
        'grad_l1_l2_ratio': l1_l2_ratio,
        'effective_dim': effective_dim,
        'grad_mean': all_grads.mean().item(),
        'grad_std': all_grads.std().item(),
    }

    if prev_gradients is not None:
        prev_grads_tensor = torch.cat(prev_gradients)
        cosine_sim = torch.nn.functional.cosine_similarity(
            current_grads_tensor.unsqueeze(0),
            prev_grads_tensor.unsqueeze(0)
        ).item()
        metrics['grad_cosine_sim'] = cosine_sim

    return metrics, current_gradients


def compute_gradient_noise(batch_grad_norms, window=50):
    """
    Compute gradient noise statistics.

    Measures mini-batch stochasticity via variance in per-batch gradient norms.

    Metrics:
    - grad_noise: Std of recent batch gradient norms
    - grad_noise_cv: Coefficient of variation (CV = σ/μ)

    Args:
        batch_grad_norms: List of per-batch gradient norms
        window: Number of recent batches to consider

    Returns:
        dict: {'grad_noise': std, 'grad_noise_cv': cv}
    """
    if not batch_grad_norms:
        return {'grad_noise': 0.0, 'grad_noise_cv': 0.0}

    recent = batch_grad_norms[-window:]
    grad_noise = np.std(recent)
    grad_mean = np.mean(recent)
    grad_noise_cv = grad_noise / (grad_mean + 1e-8)

    return {
        'grad_noise': grad_noise,
        'grad_noise_cv': grad_noise_cv,
    }


def update_ewma(history, alpha=0.1, window=10):
    """
    Compute Exponentially Weighted Moving Average (EWMA) and std.

    Approximation using recent window for simplicity.

    Args:
        history: List of values (e.g., gradient norms over epochs)
        alpha: Smoothing factor (unused in simple window approach)
        window: Number of recent values to average

    Returns:
        dict: {'ewma': moving_avg, 'ewsd': moving_std}
    """
    if not history:
        return {'ewma': 0.0, 'ewsd': 0.0}

    recent = history[-window:]
    ewma = np.mean(recent)
    ewsd = np.std(recent)

    return {'ewma': ewma, 'ewsd': ewsd}


class GradientTracker:
    """
    Stateful tracker for gradient metrics across training.

    Usage:
        tracker = GradientTracker()

        for batch in dataloader:
            loss.backward()

            if step % 10 == 0:
                tracker.update(model)

            optimizer.step()

        epoch_metrics = tracker.get_epoch_summary()
    """

    def __init__(self, track_cosine_sim=True):
        self.track_cosine_sim = track_cosine_sim
        self.prev_gradients = None
        self.batch_grad_norms = []
        self.epoch_metrics = []
        self.grad_norm_history = []

    def update(self, model):
        """
        Compute and store gradient metrics for current batch.

        Args:
            model: PyTorch model with computed gradients
        """
        metrics, new_grads = compute_gradient_metrics(
            model,
            self.prev_gradients if self.track_cosine_sim else None
        )

        if self.track_cosine_sim:
            self.prev_gradients = new_grads

        if 'grad_norm' in metrics:
            self.batch_grad_norms.append(metrics['grad_norm'])

        self.epoch_metrics.append(metrics)

    def get_epoch_summary(self):
        """
        Aggregate metrics across the epoch.

        Returns:
            dict: Averaged gradient metrics, plus noise and EWMA stats
        """
        if not self.epoch_metrics:
            return {}

        summary = {}
        keys = self.epoch_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in self.epoch_metrics if key in m]
            summary[key] = np.mean(values) if values else 0.0

        noise_metrics = compute_gradient_noise(self.batch_grad_norms)
        summary.update(noise_metrics)

        if 'grad_norm' in summary:
            self.grad_norm_history.append(summary['grad_norm'])

        ewma_metrics = update_ewma(self.grad_norm_history)
        summary.update(ewma_metrics)

        return summary

    def reset_epoch(self):
        """Clear batch-level metrics for new epoch."""
        self.epoch_metrics = []
        self.batch_grad_norms = []


# ============================================================================
# C. LAYER-LEVEL DYNAMICS
# ============================================================================

def compute_per_layer_weight_norms(model, grouped=True):
    """
    Compute L2 norm of weights for each layer.

    Args:
        model: PyTorch model
        grouped: If True, group by component (side_encoder, shared_fc, etc.)

    Returns:
        dict: {layer_name: {'norm': value, 'mean': value, 'std': value}}
              or {component: {'mean_norm': value, ...}} if grouped
    """
    stats = {}

    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            stats[name] = {
                'norm': param.data.norm(2).item(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
            }

    if grouped:
        return _group_layer_stats(stats)

    return stats


def compute_per_layer_step_sizes(model, learning_rate, grouped=True):
    """
    Compute effective step size per layer.

    Equation: Step_ℓ = η * ||g_ℓ|| / ||w_ℓ||

    Interpretation:
    - Too small → frozen layer
    - Too large → instability

    Args:
        model: PyTorch model with computed gradients
        learning_rate: Current learning rate
        grouped: If True, average by component

    Returns:
        dict: {layer_name: effective_step} or {component: {'mean_step': ...}} if grouped
    """
    layer_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            grad_norm = param.grad.norm(2).item()
            weight_norm = param.data.norm(2).item()
            effective_step = (learning_rate * grad_norm / weight_norm) if weight_norm > 0 else 0
            layer_stats[name] = effective_step

    if grouped:
        layer_groups = {
            'side_encoder': [],
            'side_aggregator': [],
            'overhead_rgb_encoder': [],
            'overhead_depth_encoder': [],
            'shared_fc': [],
            'task_heads': []
        }

        for name, step in layer_stats.items():
            if 'side_encoder' in name:
                layer_groups['side_encoder'].append(step)
            elif 'side_aggregator' in name:
                layer_groups['side_aggregator'].append(step)
            elif 'overhead_rgb_encoder' in name:
                layer_groups['overhead_rgb_encoder'].append(step)
            elif 'overhead_depth_encoder' in name:
                layer_groups['overhead_depth_encoder'].append(step)
            elif 'shared_fc' in name:
                layer_groups['shared_fc'].append(step)
            elif any(head in name for head in ['mass_head', 'calorie_head', 'macro_head']):
                layer_groups['task_heads'].append(step)

        grouped_stats = {}
        for group, steps in layer_groups.items():
            if steps:
                grouped_stats[group] = {
                    'mean_step': sum(steps) / len(steps),
                    'max_step': max(steps),
                    'min_step': min(steps)
                }

        return grouped_stats

    return layer_stats


def compute_distance_from_init(model, initial_weights, grouped=True):
    """
    Compute how far weights have moved from initialization.

    Equation: ||w_ℓ - w_{ℓ,0}|| / ||w_{ℓ,0}||

    Interpretation:
    - Small distances → early layers barely move ("lazy training")
    - Large distances → aggressive representation learning

    Args:
        model: PyTorch model
        initial_weights: dict of initial weights {name: tensor} (on CPU)
        grouped: If True, average by component

    Returns:
        dict: {layer_name: relative_distance} or {component: {'mean_dist': ...}}
    """
    distances = {}

    for name, param in model.named_parameters():
        if name in initial_weights and param.requires_grad:
            init_weight = initial_weights[name].to(param.device)
            delta = param.data - init_weight
            delta_norm = delta.norm(2).item()
            init_norm = init_weight.norm(2).item()
            rel_distance = delta_norm / init_norm if init_norm > 0 else 0
            distances[name] = rel_distance

    if grouped:
        layer_groups = {
            'side_encoder': [],
            'side_aggregator': [],
            'overhead_rgb_encoder': [],
            'overhead_depth_encoder': [],
            'shared_fc': [],
            'task_heads': [],
            'other': []
        }

        for name, dist in distances.items():
            if 'side_encoder' in name:
                layer_groups['side_encoder'].append(dist)
            elif 'side_aggregator' in name:
                layer_groups['side_aggregator'].append(dist)
            elif 'overhead_rgb_encoder' in name:
                layer_groups['overhead_rgb_encoder'].append(dist)
            elif 'overhead_depth_encoder' in name:
                layer_groups['overhead_depth_encoder'].append(dist)
            elif 'shared_fc' in name:
                layer_groups['shared_fc'].append(dist)
            elif any(head in name for head in ['mass_head', 'calorie_head', 'macro_head']):
                layer_groups['task_heads'].append(dist)
            else:
                layer_groups['other'].append(dist)

        summary = {}
        for group, dists in layer_groups.items():
            if dists:
                summary[group] = {
                    'mean_dist': sum(dists) / len(dists),
                    'max_dist': max(dists),
                    'min_dist': min(dists),
                }

        return summary

    return distances


def _group_layer_stats(layer_stats):
    """Helper to group per-layer stats by component."""
    groups = {
        'side_encoder': [],
        'side_aggregator': [],
        'overhead_rgb_encoder': [],
        'overhead_depth_encoder': [],
        'shared_fc': [],
        'task_heads': [],
        'other': []
    }

    for name, stats in layer_stats.items():
        if 'side_encoder' in name:
            groups['side_encoder'].append(stats)
        elif 'side_aggregator' in name:
            groups['side_aggregator'].append(stats)
        elif 'overhead_rgb_encoder' in name:
            groups['overhead_rgb_encoder'].append(stats)
        elif 'overhead_depth_encoder' in name:
            groups['overhead_depth_encoder'].append(stats)
        elif 'shared_fc' in name:
            groups['shared_fc'].append(stats)
        elif any(head in name for head in ['mass_head', 'calorie_head', 'macro_head']):
            groups['task_heads'].append(stats)
        else:
            groups['other'].append(stats)

    summary = {}
    for group, stats_list in groups.items():
        if stats_list:
            summary[group] = {
                'mean_norm': sum(s['norm'] for s in stats_list) / len(stats_list),
                'max_norm': max(s['norm'] for s in stats_list),
                'min_norm': min(s['norm'] for s in stats_list),
            }

    return summary


# ============================================================================
# D. ACTIVATION DYNAMICS
# ============================================================================

def compute_activation_stats(model, sample_batch, overhead_rgb=None, overhead_depth=None, sample_rate=3):
    """
    Compute activation statistics (dead ReLUs, mean/std).

    Uses forward hooks to capture intermediate activations.

    Args:
        model: PyTorch model
        sample_batch: Sample input tensor for side frames
        overhead_rgb: Optional overhead RGB input
        overhead_depth: Optional overhead depth input
        sample_rate: Only hook every Nth ReLU layer to reduce overhead

    Returns:
        dict: {
            'activation_stats': {layer_name: {'mean': ..., 'std': ..., 'dead_fraction': ...}},
            'avg_dead_fraction': float
        }
    """
    activation_stats = {}
    hooks = []

    relu_count = 0

    def hook_fn(module_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                with torch.no_grad():
                    mean_val = output.mean().item()
                    std_val = output.std().item()
                    dead_frac = (output <= 0).float().mean().item()
                    activation_stats[module_name] = {
                        'mean': mean_val,
                        'std': std_val,
                        'dead_fraction': dead_frac
                    }
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_count += 1
            if relu_count % sample_rate == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

    try:
        with torch.no_grad():
            model(sample_batch, overhead_rgb, overhead_depth)
    finally:
        for hook in hooks:
            hook.remove()

    result = {'activation_stats': activation_stats}

    if activation_stats:
        avg_dead_fraction = sum(v['dead_fraction'] for v in activation_stats.values()) / len(activation_stats)
        result['avg_dead_fraction'] = avg_dead_fraction
    else:
        result['avg_dead_fraction'] = 0.0

    return result


# ============================================================================
# E. TIMING & SYSTEM PERFORMANCE
# ============================================================================

class TimingTracker:
    """
    Context-managed timing tracker for training loops.

    Usage:
        timer = TimingTracker()

        timer.start_epoch()
        for batch in dataloader:
            with timer.time('data_load'):
                batch = load_data()

            with timer.time('forward'):
                pred = model(batch)

            with timer.time('backward'):
                loss.backward()
                optimizer.step()

        metrics = timer.end_epoch(num_samples=len(dataset))
    """

    def __init__(self):
        self.timings = {}
        self.epoch_start_time = None
        self.epoch_end_time = None

    def start_epoch(self):
        """Mark the start of an epoch."""
        self.epoch_start_time = time.time()
        self.timings = {}

    def time(self, name):
        """
        Context manager for timing a specific operation.

        Args:
            name: Name of the operation (e.g., 'data_load', 'forward', 'backward')

        Returns:
            Context manager
        """
        return _TimingContext(self, name)

    def _record(self, name, duration):
        """Internal: record a timing."""
        if name not in self.timings:
            self.timings[name] = 0.0
        self.timings[name] += duration

    def end_epoch(self, num_samples=None):
        """
        Finalize epoch timing and compute metrics.

        Args:
            num_samples: Number of samples processed (for throughput calculation)

        Returns:
            dict: Timing metrics
        """
        self.epoch_end_time = time.time()
        epoch_time = self.epoch_end_time - self.epoch_start_time

        metrics = {
            'epoch_time': epoch_time,
            **self.timings
        }

        if num_samples is not None and epoch_time > 0:
            metrics['throughput'] = num_samples / epoch_time

        return metrics

    def get_avg_times(self, num_batches):
        """
        Get average time per batch for each operation.

        Args:
            num_batches: Number of batches processed

        Returns:
            dict: {operation: avg_time_per_batch}
        """
        if num_batches == 0:
            return {}

        return {name: total_time / num_batches for name, total_time in self.timings.items()}


class _TimingContext:
    """Helper context manager for timing operations."""

    def __init__(self, tracker, name):
        self.tracker = tracker
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.tracker._record(self.name, duration)
        return False
