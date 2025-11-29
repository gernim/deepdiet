"""
TensorBoard Logger

Centralized logging interface for all metrics to TensorBoard.
"""
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json


class TensorBoardLogger:
    """
    TensorBoard logger for DeepDiet training.

    Automatically organizes metrics by category and provides convenience methods.

    Usage:
        logger = TensorBoardLogger(log_dir='runs/experiment_1')

        # Log scalars
        logger.log_task_metrics(epoch, train_metrics, prefix='train')
        logger.log_gradient_metrics(epoch, grad_metrics)
        logger.log_layer_metrics(epoch, layer_metrics)

        # Log custom scalar
        logger.log_scalar('learning_rate', lr, epoch)

        logger.close()
    """

    def __init__(self, log_dir='runs', experiment_name=None, config=None, flush_secs=30):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Optional experiment name (creates subdirectory)
            config: Optional dict of hyperparameters to log
            flush_secs: How often to flush to disk
        """
        if experiment_name:
            log_path = Path(log_dir) / experiment_name
        else:
            log_path = Path(log_dir)

        self.log_dir = log_path
        self.writer = SummaryWriter(str(log_path), flush_secs=flush_secs)

        # Log hyperparameters
        if config:
            self.log_hparams(config)

        print(f"TensorBoard logging to: {log_path}")
        print(f"  View with: tensorboard --logdir {log_dir}")

    def log_scalar(self, tag, value, step):
        """Log a single scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag_dict, step, prefix=None):
        """
        Log multiple scalars at once.

        Args:
            tag_dict: Dict of {name: value}
            step: Global step (usually epoch number)
            prefix: Optional prefix for all tags (e.g., 'train/', 'val/')
        """
        for name, value in tag_dict.items():
            full_tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(full_tag, value, step)

    def log_task_metrics(self, step, metrics, prefix='train'):
        """
        Log task performance metrics.

        Expects metrics like: {
            'cal_mae': ...,
            'mass_mae': ...,
            'cal_relative_mae': ...,
            ...
        }

        Organizes into TensorBoard groups:
        - Task/MAE/{prefix}/cal
        - Task/MAE/{prefix}/mass
        - Task/Relative_MAE/{prefix}/cal
        ...
        """
        # Group by metric type
        mae_metrics = {}
        relative_mae_metrics = {}
        median_ae_metrics = {}
        signed_error_metrics = {}

        for name, value in metrics.items():
            if '_mae' in name and 'relative' not in name:
                task_name = name.replace('_mae', '')
                mae_metrics[task_name] = value
            elif '_relative_mae' in name:
                task_name = name.replace('_relative_mae', '')
                relative_mae_metrics[task_name] = value
            elif '_median_ae' in name:
                task_name = name.replace('_median_ae', '')
                median_ae_metrics[task_name] = value
            elif '_signed_error' in name:
                task_name = name.replace('_signed_error', '')
                signed_error_metrics[task_name] = value

        # Log grouped metrics
        if mae_metrics:
            self.writer.add_scalars(f'Task/MAE/{prefix}', mae_metrics, step)
        if relative_mae_metrics:
            self.writer.add_scalars(f'Task/Relative_MAE/{prefix}', relative_mae_metrics, step)
        if median_ae_metrics:
            self.writer.add_scalars(f'Task/Median_AE/{prefix}', median_ae_metrics, step)
        if signed_error_metrics:
            self.writer.add_scalars(f'Task/Signed_Error/{prefix}', signed_error_metrics, step)

    def log_gradient_metrics(self, step, metrics):
        """
        Log gradient metrics.

        Expects metrics like: {
            'grad_norm': ...,
            'grad_l2': ...,
            'grad_cosine_sim': ...,
            ...
        }

        Organizes into:
        - Gradients/Norms/grad_norm
        - Gradients/Norms/grad_l2
        - Gradients/Stability/cosine_sim
        - Gradients/Noise/grad_noise
        ...
        """
        norm_metrics = {
            'grad_norm': metrics.get('grad_norm'),
            'grad_l2': metrics.get('grad_l2'),
        }
        norm_metrics = {k: v for k, v in norm_metrics.items() if v is not None}
        if norm_metrics:
            self.writer.add_scalars('Gradients/Norms', norm_metrics, step)

        # Stability metrics
        stability_metrics = {
            'cosine_sim': metrics.get('grad_cosine_sim'),
            'l1_l2_ratio': metrics.get('grad_l1_l2_ratio'),
        }
        stability_metrics = {k: v for k, v in stability_metrics.items() if v is not None}
        if stability_metrics:
            self.writer.add_scalars('Gradients/Stability', stability_metrics, step)

        # Noise metrics
        noise_metrics = {
            'grad_noise': metrics.get('grad_noise'),
            'grad_noise_cv': metrics.get('grad_noise_cv'),
        }
        noise_metrics = {k: v for k, v in noise_metrics.items() if v is not None}
        if noise_metrics:
            self.writer.add_scalars('Gradients/Noise', noise_metrics, step)

        # EWMA metrics
        ewma_metrics = {
            'ewma': metrics.get('ewma'),
            'ewsd': metrics.get('ewsd'),
        }
        ewma_metrics = {k: v for k, v in ewma_metrics.items() if v is not None}
        if ewma_metrics:
            self.writer.add_scalars('Gradients/EWMA', ewma_metrics, step)

        # Effective dimension
        if 'effective_dim' in metrics:
            self.writer.add_scalar('Gradients/Effective_Dimension', metrics['effective_dim'], step)

    def log_layer_metrics(self, step, metrics, metric_type='step_size'):
        """
        Log per-layer metrics.

        Args:
            step: Global step
            metrics: Dict of {component: {'mean_step': ..., 'max_step': ...}} or similar
            metric_type: 'step_size', 'distance_from_init', or 'weight_norm'
        """
        mean_metrics = {}
        max_metrics = {}

        for component, stats in metrics.items():
            if 'mean_step' in stats:
                mean_metrics[component] = stats['mean_step']
            elif 'mean_dist' in stats:
                mean_metrics[component] = stats['mean_dist']
            elif 'mean_norm' in stats:
                mean_metrics[component] = stats['mean_norm']

            if 'max_step' in stats:
                max_metrics[component] = stats['max_step']
            elif 'max_dist' in stats:
                max_metrics[component] = stats['max_dist']
            elif 'max_norm' in stats:
                max_metrics[component] = stats['max_norm']

        if mean_metrics:
            tag = f'Layers/{metric_type}/mean'
            self.writer.add_scalars(tag, mean_metrics, step)

        if max_metrics:
            tag = f'Layers/{metric_type}/max'
            self.writer.add_scalars(tag, max_metrics, step)

    def log_activation_metrics(self, step, metrics):
        """
        Log activation statistics.

        Expects metrics like: {
            'avg_dead_fraction': ...,
            'activation_stats': {layer_name: {'mean': ..., 'std': ..., 'dead_fraction': ...}}
        }
        """
        if 'avg_dead_fraction' in metrics:
            self.writer.add_scalar('Activations/Avg_Dead_Fraction', metrics['avg_dead_fraction'], step)

        # Optionally log per-layer activation stats (can be verbose)
        if 'activation_stats' in metrics:
            dead_fractions = {name: stats['dead_fraction']
                            for name, stats in metrics['activation_stats'].items()}
            if dead_fractions:
                # Just log a few key layers to avoid clutter
                # Or aggregate by component
                pass

    def log_timing_metrics(self, step, metrics):
        """
        Log timing and throughput metrics.

        Expects metrics like: {
            'epoch_time': ...,
            'data_load': ...,
            'forward': ...,
            'backward': ...,
            'throughput': ...
        }
        """
        timing_metrics = {}
        for key in ['epoch_time', 'data_load', 'forward', 'backward']:
            if key in metrics:
                timing_metrics[key] = metrics[key]

        if timing_metrics:
            self.writer.add_scalars('Timing/Breakdown', timing_metrics, step)

        if 'throughput' in metrics:
            self.writer.add_scalar('Timing/Throughput', metrics['throughput'], step)

    def log_loss(self, step, total_loss, task_losses=None, prefix='train'):
        """
        Log total loss and per-task losses.

        Args:
            step: Global step
            total_loss: Scalar total loss
            task_losses: Optional dict of {task_name: loss}
            prefix: 'train' or 'val'
        """
        self.writer.add_scalar(f'Loss/{prefix}/total', total_loss, step)

        if task_losses:
            self.writer.add_scalars(f'Loss/{prefix}/per_task', task_losses, step)

    def log_hparams(self, config, metrics=None):
        """
        Log hyperparameters.

        Args:
            config: Dict of hyperparameters
            metrics: Optional dict of metrics to associate with hparams
        """
        # Convert all values to serializable types
        clean_config = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool)):
                clean_config[k] = v
            else:
                clean_config[k] = str(v)

        self.writer.add_hparams(clean_config, metrics or {})

        # Also save as JSON
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(clean_config, f, indent=2)

    def flush(self):
        """Force write all pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the logger and flush all pending writes."""
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
