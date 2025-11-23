#!/usr/bin/env python3
"""
Multi-view training script for DeepDiet.
Supports side angle images only, or side angles + overhead RGB/depth.
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision.io')

from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import math
import argparse
import io
import time
from collections import defaultdict
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "nutrition5k_dataset"
OVERHEAD_TRAIN_CSV = REPO / "indexes" / "overhead_train.csv"
OVERHEAD_TEST_CSV = REPO / "indexes" / "overhead_test.csv"
SIDE_TRAIN_CSV = REPO / "indexes" / "side_frames_train.csv"
SIDE_TEST_CSV = REPO / "indexes" / "side_frames_test.csv"
TARGETS = ["cal", "mass", "fat", "carb", "protein"]

class MultiViewDataset(Dataset):
    """Dataset that uses side angle views, optionally with overhead RGB/depth.

    Supports both local filesystem and GCS streaming.
    """

    def __init__(self, overhead_csv, side_csv, data_root, train=True, max_side_frames=16,
                 use_overhead=False, use_gcs=False, gcs_bucket=None, gcs_prefix=None,
                 image_size=256, cache_dir=None):
        """
        Args:
            overhead_csv: Path to overhead images CSV (can be None if use_overhead=False)
            side_csv: Path to side angle frames CSV
            data_root: Root directory for data (ignored if use_gcs=True)
            train: Training mode (enables augmentation)
            max_side_frames: Maximum number of side frames to use per dish
            use_overhead: Whether to load overhead RGB/depth images
            use_gcs: If True, stream images from GCS instead of local filesystem
            gcs_bucket: GCS bucket name (required if use_gcs=True)
            gcs_prefix: Path prefix in GCS bucket (required if use_gcs=True)
            image_size: Target image size (default 256, use 128 or 192 for faster training)
            cache_dir: Optional directory to cache GCS images locally
        """
        self.data_root = Path(data_root) if not use_gcs else None
        self.train = train
        self.max_side_frames = max_side_frames
        self.use_overhead = use_overhead
        self.use_gcs = use_gcs
        self.image_size = image_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Setup GCS client if needed
        if use_gcs:
            from google.cloud import storage
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(gcs_bucket)
            self.gcs_prefix = gcs_prefix
            print(f"Using GCS bucket: gs://{gcs_bucket}/{gcs_prefix}")

        # Load side angle data
        side_df = pd.read_csv(side_csv)

        # Group side frames by dish_id and get targets
        side_grouped = side_df.groupby('dish_id')
        self.side_frames = side_grouped['image'].apply(list).to_dict()

        # Get unique dishes with their targets (take first row per dish since targets are the same)
        dish_data = side_df.groupby('dish_id')[TARGETS].first().reset_index()

        if use_overhead:
            # Load overhead data and merge
            overhead_df = pd.read_csv(overhead_csv)
            overhead_df = overhead_df[["dish_id", "rgb", "depth_raw"] + TARGETS].dropna()

            # Only keep dishes that have both overhead and side angle images
            self.df = dish_data.merge(overhead_df[["dish_id", "rgb", "depth_raw"]], on='dish_id', how='inner')
            print(f"Loaded {len(self.df)} dishes with both overhead and side views")
        else:
            # Use only side angle data
            self.df = dish_data[dish_data['dish_id'].isin(self.side_frames.keys())].reset_index(drop=True)
            print(f"Loaded {len(self.df)} dishes with side angle views only")

        # Calculate resize dimensions (slightly larger for cropping)
        resize_size = int(image_size * 1.125)  # 12.5% larger for crop

        # Transforms for side view RGB images (center crop to keep dish in frame)
        if train:
            self.side_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),  # Center crop to focus on dish
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.side_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])

        # Overhead RGB transform (always center crop - dish is centered)
        self.overhead_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])

        # Depth transform (always center crop)
        self.depth_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def _load_image(self, image_path):
        """Load image from either local filesystem or GCS with optional caching."""
        if self.use_gcs:
            # Check cache first if enabled
            if hasattr(self, 'cache_dir') and self.cache_dir is not None:
                cache_path = self.cache_dir / image_path
                if cache_path.exists():
                    return Image.open(cache_path)

                # Load from GCS and cache
                blob_path = self.gcs_prefix + image_path
                blob = self.bucket.blob(blob_path)
                image_bytes = blob.download_as_bytes()

                # Save to cache
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    f.write(image_bytes)

                return Image.open(io.BytesIO(image_bytes))
            else:
                # Load from GCS without caching
                blob_path = self.gcs_prefix + image_path
                blob = self.bucket.blob(blob_path)
                image_bytes = blob.download_as_bytes()
                return Image.open(io.BytesIO(image_bytes))
        else:
            # Load from local filesystem
            return Image.open(self.data_root / image_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']

        # Load overhead RGB and depth if using overhead
        if self.use_overhead:
            overhead_rgb = self._load_image(row['rgb']).convert('RGB')
            overhead_rgb = self.overhead_transform(overhead_rgb)  # Use overhead-specific transform

            # Load overhead depth (if available)
            try:
                overhead_depth = self._load_image(row['depth_raw']).convert('L')
                overhead_depth = self.depth_transform(overhead_depth)
            except:
                # If depth not available, use zeros
                overhead_depth = torch.zeros(1, self.image_size, self.image_size)
        else:
            overhead_rgb = None
            overhead_depth = None

        # Load side angle frames
        side_frame_paths = self.side_frames[dish_id][:self.max_side_frames]
        side_frames = []
        for frame_path in side_frame_paths:
            try:
                frame = self._load_image(frame_path).convert('RGB')
                frame = self.side_transform(frame)  # Use side-specific transform
                side_frames.append(frame)
            except Exception as e:
                # Skip failed frames
                pass

        # Pad or truncate to max_side_frames
        while len(side_frames) < self.max_side_frames:
            side_frames.append(torch.zeros(3, self.image_size, self.image_size))
        side_frames = torch.stack(side_frames[:self.max_side_frames])

        # Targets
        targets = torch.tensor([row[t] for t in TARGETS], dtype=torch.float32)

        result = {
            'side_frames': side_frames,
            'targets': targets,
            'dish_id': dish_id
        }

        if self.use_overhead:
            result['overhead_rgb'] = overhead_rgb
            result['overhead_depth'] = overhead_depth

        return result


class MultiViewModel(nn.Module):
    """Multi-view model with EfficientNet-B3 backbone and multi-task heads.

    Architecture inspired by Nutrition5k paper but with multi-view side angles.
    Uses separate task-specific heads for mass, calories, and macronutrients.
    """

    def __init__(self, num_side_frames=16, use_overhead=False, use_depth=True, chunk_size=4):
        super().__init__()
        self.use_overhead = use_overhead
        self.use_depth = use_depth and use_overhead
        self.chunk_size = chunk_size

        # Side angle encoder - EfficientNet-B0 (smaller, memory efficient)
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.side_encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        side_feat_dim = self.side_encoder.classifier[1].in_features
        self.side_encoder.classifier = nn.Identity()  # Remove classification head

        # Aggregation for side frames - BiLSTM
        lstm_hidden = 384  # Reduced from 768 to save memory
        self.side_aggregator = nn.LSTM(side_feat_dim, lstm_hidden, batch_first=True, bidirectional=True)
        aggregated_feat_dim = lstm_hidden * 2  # bidirectional

        total_feat_dim = aggregated_feat_dim

        # Overhead encoders (if using overhead)
        if use_overhead:
            # Overhead RGB encoder - EfficientNet-B0
            self.overhead_rgb_encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            overhead_feat_dim = self.overhead_rgb_encoder.classifier[1].in_features
            self.overhead_rgb_encoder.classifier = nn.Identity()
            total_feat_dim += overhead_feat_dim

            # Overhead depth encoder (if using depth)
            if self.use_depth:
                self.overhead_depth_encoder = efficientnet_b0(weights=None)
                # Modify first conv for single channel depth
                original_conv = self.overhead_depth_encoder.features[0][0]
                self.overhead_depth_encoder.features[0][0] = nn.Conv2d(
                    1, original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False
                )
                depth_feat_dim = self.overhead_depth_encoder.classifier[1].in_features
                self.overhead_depth_encoder.classifier = nn.Identity()
                total_feat_dim += depth_feat_dim

        # Shared fusion layers (reduced size to save memory)
        self.shared_fc = nn.Sequential(
            nn.Linear(total_feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Task-specific heads (following Nutrition5k paper structure)
        # Mass prediction head
        self.mass_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Calorie prediction head
        self.calorie_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Macronutrient prediction head (fat, carb, protein)
        self.macro_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)  # fat, carb, protein
        )

    def forward(self, side_frames, overhead_rgb=None, overhead_depth=None):
        batch_size = side_frames.size(0)

        # Encode side frames with chunking to reduce peak memory
        num_frames = side_frames.size(1)
        side_frames_flat = side_frames.view(batch_size * num_frames, 3, 256, 256)

        # Process frames in smaller chunks to avoid OOM
        side_feats = []
        for i in range(0, side_frames_flat.size(0), self.chunk_size):
            chunk = side_frames_flat[i:i + self.chunk_size]
            chunk_feat = self.side_encoder(chunk)
            side_feats.append(chunk_feat)

        side_feat = torch.cat(side_feats, dim=0)
        side_feat = side_feat.view(batch_size, num_frames, -1)

        # Aggregate side features with BiLSTM
        side_feat, _ = self.side_aggregator(side_feat)
        side_feat = side_feat[:, -1, :]  # Take last hidden state

        # Collect features
        features = [side_feat]

        if self.use_overhead and overhead_rgb is not None:
            overhead_rgb_feat = self.overhead_rgb_encoder(overhead_rgb)
            features.append(overhead_rgb_feat)

            if self.use_depth and overhead_depth is not None:
                overhead_depth_feat = self.overhead_depth_encoder(overhead_depth)
                features.append(overhead_depth_feat)

        # Fuse all features
        combined = torch.cat(features, dim=1)

        # Shared layers
        shared_repr = self.shared_fc(combined)

        # Task-specific predictions
        mass = self.mass_head(shared_repr)
        calories = self.calorie_head(shared_repr)
        macros = self.macro_head(shared_repr)  # [fat, carb, protein]

        # Concatenate outputs in order: [cal, mass, fat, carb, protein]
        output = torch.cat([calories, mass, macros], dim=1)

        return output


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mae_metric(pred, target):
    return (pred - target).abs().mean().item()


def compute_gradient_metrics(model, prev_gradients=None, learning_rate=1e-4):
    """Compute various gradient statistics for monitoring training."""
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
            grad_list.append(grad.flatten())
            current_gradients.append(grad.flatten().clone())

            total_norm += grad.norm(2).item() ** 2
            total_grad_sum += grad.sum().item()
            total_grad_sq += (grad ** 2).sum().item()
            total_abs_grad += grad.abs().sum().item()
            num_params += grad.numel()

    if num_params == 0:
        return {}, None

    # Concatenate all gradients
    all_grads = torch.cat(grad_list)
    current_grads_tensor = torch.cat(current_gradients)

    # Compute metrics
    grad_norm = math.sqrt(total_norm)  # L2 norm
    l2_metric = math.sqrt(total_grad_sq / num_params)  # L2
    l1_l2_ratio = total_abs_grad / (l2_metric * math.sqrt(num_params)) if l2_metric > 0 else 0

    # Effective dimension
    effective_dim = (total_grad_sum ** 2) / total_grad_sq if total_grad_sq > 0 else 0

    metrics = {
        'grad_norm': grad_norm,
        'grad_l2': l2_metric,
        'grad_l1_l2_ratio': l1_l2_ratio,
        'effective_dim': effective_dim,
        'grad_mean': all_grads.mean().item(),
        'grad_std': all_grads.std().item(),
    }

    # Gradient cosine similarity (if we have previous gradients)
    if prev_gradients is not None:
        prev_grads_tensor = torch.cat(prev_gradients)
        cosine_sim = torch.nn.functional.cosine_similarity(
            current_grads_tensor.unsqueeze(0),
            prev_grads_tensor.unsqueeze(0)
        ).item()
        metrics['grad_cosine_sim'] = cosine_sim

    return metrics, current_gradients


def compute_per_layer_step_sizes(model, learning_rate):
    """Compute effective step size per layer group."""
    layer_stats = {}

    # Group by major components
    layer_groups = {
        'side_encoder': [],
        'side_aggregator': [],
        'shared_fc': [],
        'task_heads': []
    }

    for name, param in model.named_parameters():
        if param.grad is not None:
            # Compute effective step size: lr * ||g|| / ||w||
            grad_norm = param.grad.norm(2).item()
            weight_norm = param.data.norm(2).item()
            effective_step = (learning_rate * grad_norm / weight_norm) if weight_norm > 0 else 0

            # Classify into groups
            if 'side_encoder' in name:
                layer_groups['side_encoder'].append(effective_step)
            elif 'side_aggregator' in name:
                layer_groups['side_aggregator'].append(effective_step)
            elif 'shared_fc' in name:
                layer_groups['shared_fc'].append(effective_step)
            elif any(head in name for head in ['mass_head', 'calorie_head', 'macro_head']):
                layer_groups['task_heads'].append(effective_step)

    # Average per group
    for group, steps in layer_groups.items():
        if steps:
            layer_stats[group] = {
                'mean_step': sum(steps) / len(steps),
                'max_step': max(steps),
                'min_step': min(steps)
            }

    return layer_stats


def compute_activation_stats(model, side_frames, overhead_rgb=None, overhead_depth=None):
    """Compute activation statistics (dead ReLUs, mean/std) - lightweight version."""
    activation_stats = {}
    hooks = []

    relu_count = 0

    def hook_fn(module_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # Only track key statistics to minimize overhead
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

    # Register hooks only on key ReLU layers (not all to reduce overhead)
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_count += 1
            # Sample every 3rd ReLU to reduce overhead
            if relu_count % 3 == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

    # Forward pass
    try:
        with torch.no_grad():
            model(side_frames, overhead_rgb, overhead_depth)
    finally:
        # Always remove hooks
        for hook in hooks:
            hook.remove()

    return activation_stats


def compute_distance_from_init(model, initial_weights):
    """Compute how far weights have moved from initialization."""
    distances = {}

    for name, param in model.named_parameters():
        if name in initial_weights and param.requires_grad:
            init_weight = initial_weights[name]
            # Compute relative distance: ||w - w0|| / ||w0||
            delta = param.data - init_weight
            delta_norm = delta.norm(2).item()
            init_norm = init_weight.norm(2).item()
            rel_distance = delta_norm / init_norm if init_norm > 0 else 0

            # Group by component
            if 'side_encoder' in name:
                group = 'side_encoder'
            elif 'side_aggregator' in name:
                group = 'side_aggregator'
            elif 'shared_fc' in name:
                group = 'shared_fc'
            elif any(head in name for head in ['mass_head', 'calorie_head', 'macro_head']):
                group = 'task_heads'
            else:
                group = 'other'

            if group not in distances:
                distances[group] = []
            distances[group].append(rel_distance)

    # Average per group
    summary = {}
    for group, dists in distances.items():
        if dists:
            summary[group] = {
                'mean_dist': sum(dists) / len(dists),
                'max_dist': max(dists)
            }

    return summary


def compute_weight_stats(model):
    """Compute weight statistics per layer."""
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'weight' in name:
            stats[name] = {
                'norm': param.data.norm(2).item(),
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
            }
    return stats


def main():
    parser = argparse.ArgumentParser(description='Train DeepDiet multi-view model')
    parser.add_argument('--use-overhead', action='store_true',
                        help='Use overhead RGB and depth images (default: side angles only)')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth images when using overhead (only works with --use-overhead)')
    parser.add_argument('--use-gcs', action='store_true',
                        help='Stream images from GCS bucket instead of local filesystem')
    parser.add_argument('--gcs-bucket', type=str, default='deepdiet-dataset',
                        help='GCS bucket name (default: deepdiet-dataset)')
    parser.add_argument('--gcs-prefix', type=str, default='nutrition5k_dataset/',
                        help='Path prefix in GCS bucket (default: nutrition5k_dataset/)')
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
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image size for training (default: 256, use 128 or 192 for faster training)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache GCS images locally (speeds up subsequent epochs)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from (e.g., indexes/side_angles_best.pt)')
    parser.add_argument('--train-csv', type=str, default=None,
                        help='Path to training CSV file (default: use built-in paths based on mode)')
    parser.add_argument('--test-csv', type=str, default=None,
                        help='Path to test CSV file (default: use built-in paths based on mode)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Data source: {'GCS bucket' if args.use_gcs else 'Local filesystem'}")
    print(f"Mode: {'Side angles + overhead RGB/depth' if args.use_overhead else 'Side angles only'}")
    if args.use_overhead and args.no_depth:
        print("  (depth disabled)")

    # Determine CSV paths
    if args.train_csv:
        train_side_csv = Path(args.train_csv)
        # For multimodal training, look for corresponding overhead CSV
        if args.use_overhead and 'multimodal_side' in args.train_csv:
            train_overhead_csv = Path(args.train_csv.replace('multimodal_side', 'multimodal_overhead'))
        elif args.use_overhead:
            train_overhead_csv = OVERHEAD_TRAIN_CSV
        else:
            train_overhead_csv = None
    else:
        train_side_csv = SIDE_TRAIN_CSV
        train_overhead_csv = OVERHEAD_TRAIN_CSV if args.use_overhead else None

    if args.test_csv:
        test_side_csv = Path(args.test_csv)
        # For multimodal training, look for corresponding overhead CSV
        if args.use_overhead and 'multimodal_side' in args.test_csv:
            test_overhead_csv = Path(args.test_csv.replace('multimodal_side', 'multimodal_overhead'))
        elif args.use_overhead:
            test_overhead_csv = OVERHEAD_TEST_CSV
        else:
            test_overhead_csv = None
    else:
        test_side_csv = SIDE_TEST_CSV
        test_overhead_csv = OVERHEAD_TEST_CSV if args.use_overhead else None

    # Create datasets
    print("\nLoading datasets...")
    train_ds = MultiViewDataset(
        train_overhead_csv,
        train_side_csv,
        DATA_ROOT,
        train=True,
        max_side_frames=args.max_frames,
        use_overhead=args.use_overhead,
        use_gcs=args.use_gcs,
        gcs_bucket=args.gcs_bucket if args.use_gcs else None,
        gcs_prefix=args.gcs_prefix if args.use_gcs else None,
        image_size=args.image_size,
        cache_dir=args.cache_dir
    )
    val_ds = MultiViewDataset(
        test_overhead_csv,
        test_side_csv,
        DATA_ROOT,
        train=False,
        max_side_frames=args.max_frames,
        use_overhead=args.use_overhead,
        use_gcs=args.use_gcs,
        gcs_bucket=args.gcs_bucket if args.use_gcs else None,
        gcs_prefix=args.gcs_prefix if args.use_gcs else None,
        image_size=args.image_size,
        cache_dir=args.cache_dir
    )

    # Note: Use num_workers=0 when streaming from GCS to avoid authentication issues in workers
    num_workers = 0 if args.use_gcs else 2
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    # Create model
    print("\nInitializing model...")
    model = MultiViewModel(
        num_side_frames=args.max_frames,
        use_overhead=args.use_overhead,
        use_depth=not args.no_depth,
        chunk_size=args.chunk_size
    ).to(device)

    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✓ Resumed from checkpoint: {args.resume}")
        else:
            print(f"⚠ Checkpoint not found: {args.resume}, starting from scratch")

    # Multi-task MAE loss (following Nutrition5k paper)
    criterion = nn.L1Loss(reduction='none')  # Per-element MAE
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Use automatic mixed precision to reduce memory usage
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Task weights (can be tuned)
    task_weights = {
        'cal': 1.0,
        'mass': 1.0,
        'fat': 1.0,
        'carb': 1.0,
        'protein': 1.0
    }

    # Store initial weights for distance tracking
    print("Saving initial weights...")
    initial_weights = {name: param.data.clone().cpu() for name, param in model.named_parameters() if param.requires_grad}

    # Training loop
    best_val_loss = math.inf

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Loss: Multi-task MAE (cal, mass, fat, carb, protein)")
    print("=" * 70)

    # Tracking for EWMA gradient metrics
    grad_norm_history = []
    prev_gradients = None
    batch_grad_norms = []  # Track per-batch gradient norms for noise estimation

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # Train
            model.train()
            train_losses = {task: 0.0 for task in TARGETS}
            train_total_loss = 0.0

            # Timing trackers
            data_load_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            batch_count = 0
            grad_metrics_accum = defaultdict(float)

            batch_start = time.time()
            # Add progress bar to monitor batches
            train_pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
            for batch in train_pbar:
                data_load_time += time.time() - batch_start
                batch_count += 1

                forward_start = time.time()
                side_frames = batch['side_frames'].to(device)
                targets = batch['targets'].to(device)

                overhead_rgb = batch.get('overhead_rgb')
                overhead_depth = batch.get('overhead_depth')

                if overhead_rgb is not None:
                    overhead_rgb = overhead_rgb.to(device)
                if overhead_depth is not None:
                    overhead_depth = overhead_depth.to(device)

                optimizer.zero_grad()

                # Use mixed precision if available
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        pred = model(side_frames, overhead_rgb, overhead_depth)
                        forward_time += time.time() - forward_start
                        # Compute per-task MAE loss
                        task_losses = criterion(pred, targets).mean(dim=0)  # [5] losses
                        # Weighted sum of task losses
                        weighted_loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))

                    backward_start = time.time()
                    scaler.scale(weighted_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Compute gradient metrics periodically
                    current_lr = optimizer.param_groups[0]['lr']
                    if batch_count % 10 == 0:
                        grad_metrics, new_grads = compute_gradient_metrics(model, prev_gradients, current_lr)
                        for k, v in grad_metrics.items():
                            grad_metrics_accum[k] += v
                        prev_gradients = new_grads

                    # Track batch gradient norm for noise estimation
                    batch_grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    batch_grad_norms.append(batch_grad_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    backward_time += time.time() - backward_start
                else:
                    pred = model(side_frames, overhead_rgb, overhead_depth)
                    forward_time += time.time() - forward_start
                    # Compute per-task MAE loss
                    task_losses = criterion(pred, targets).mean(dim=0)  # [5] losses
                    # Weighted sum of task losses
                    weighted_loss = sum(task_losses[i] * task_weights[TARGETS[i]] for i in range(len(TARGETS)))

                    backward_start = time.time()
                    weighted_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    # Compute gradient metrics periodically
                    current_lr = optimizer.param_groups[0]['lr']
                    if batch_count % 10 == 0:
                        grad_metrics, new_grads = compute_gradient_metrics(model, prev_gradients, current_lr)
                        for k, v in grad_metrics.items():
                            grad_metrics_accum[k] += v
                        prev_gradients = new_grads

                    # Track batch gradient norm for noise estimation
                    batch_grad_norm = sum(p.grad.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    batch_grad_norms.append(batch_grad_norm)

                    optimizer.step()
                    backward_time += time.time() - backward_start

                # Track losses
                batch_size = side_frames.size(0)
                train_total_loss += weighted_loss.item() * batch_size
                for i, task in enumerate(TARGETS):
                    train_losses[task] += task_losses[i].item() * batch_size

                # Update progress bar with timing info
                train_pbar.set_postfix({
                    'loss': f'{weighted_loss.item():.2f}',
                    'data': f'{data_load_time/batch_count:.2f}s',
                    'fwd': f'{forward_time/batch_count:.2f}s',
                    'bwd': f'{backward_time/batch_count:.2f}s'
                })

                batch_start = time.time()

        # Average losses
        train_total_loss /= len(train_ds)
        for task in TARGETS:
            train_losses[task] /= len(train_ds)

        # Validate
        model.eval()
        val_losses = {task: 0.0 for task in TARGETS}
        val_total_loss = 0.0

        with torch.no_grad():
            for batch in val_dl:
                side_frames = batch['side_frames'].to(device)
                targets = batch['targets'].to(device)

                overhead_rgb = batch.get('overhead_rgb')
                overhead_depth = batch.get('overhead_depth')

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
                batch_size = side_frames.size(0)
                val_total_loss += weighted_loss.item() * batch_size
                for i, task in enumerate(TARGETS):
                    val_losses[task] += task_losses[i].item() * batch_size

            # Average losses
            val_total_loss /= len(val_ds)
            for task in TARGETS:
                val_losses[task] /= len(val_ds)

            epoch_time = time.time() - epoch_start

            # Average gradient metrics
            num_grad_samples = max(1, batch_count // 10)
            for k in grad_metrics_accum:
                grad_metrics_accum[k] /= num_grad_samples

            # Track gradient norm history for EWMA
            if 'grad_norm' in grad_metrics_accum:
                grad_norm_history.append(grad_metrics_accum['grad_norm'])

            # Print epoch summary
            print(f"\n[Epoch {epoch:02d}/{args.epochs}] Time: {epoch_time:.1f}s (data: {data_load_time:.1f}s, fwd: {forward_time:.1f}s, bwd: {backward_time:.1f}s)")
            print(f"  Total Loss: {train_total_loss:.3f} (train) / {val_total_loss:.3f} (val)")
            print(f"  Cal:     {train_losses['cal']:7.2f} / {val_losses['cal']:7.2f}")
            print(f"  Mass:    {train_losses['mass']:7.2f} / {val_losses['mass']:7.2f}")
            print(f"  Fat:     {train_losses['fat']:7.2f} / {val_losses['fat']:7.2f}")
            print(f"  Carb:    {train_losses['carb']:7.2f} / {val_losses['carb']:7.2f}")
            print(f"  Protein: {train_losses['protein']:7.2f} / {val_losses['protein']:7.2f}")

            # Compute additional metrics (do this every few epochs to minimize overhead)
            if epoch % 2 == 0:
                # Gradient noise
                if batch_grad_norms:
                    import numpy as np
                    grad_noise = np.std(batch_grad_norms[-50:]) if len(batch_grad_norms) >= 50 else np.std(batch_grad_norms)
                    grad_noise_cv = grad_noise / (np.mean(batch_grad_norms[-50:]) + 1e-8) if len(batch_grad_norms) >= 50 else 0
                else:
                    grad_noise = 0
                    grad_noise_cv = 0

                # Per-layer step sizes
                current_lr = optimizer.param_groups[0]['lr']
                layer_steps = compute_per_layer_step_sizes(model, current_lr)

                # Distance from initialization
                dist_from_init = compute_distance_from_init(model, initial_weights)

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
                    activation_stats = compute_activation_stats(model, val_side_frames, val_overhead_rgb, val_overhead_depth)
                    model.train()

                    # Aggregate activation stats
                    if activation_stats:
                        avg_dead_fraction = np.mean([v['dead_fraction'] for v in activation_stats.values()])
                    else:
                        avg_dead_fraction = 0
                except:
                    avg_dead_fraction = 0
                    layer_steps = {}
                    dist_from_init = {}

            # Print gradient metrics
            if grad_metrics_accum:
                print(f"  Grad norm: {grad_metrics_accum.get('grad_norm', 0):.4f}  |  " +
                      f"L2: {grad_metrics_accum.get('grad_l2', 0):.6f}  |  " +
                      f"Effective dim: {grad_metrics_accum.get('effective_dim', 0):.2f}")

                if 'grad_cosine_sim' in grad_metrics_accum:
                    print(f"  Cosine sim: {grad_metrics_accum['grad_cosine_sim']:.4f}  |  " +
                          f"(1.0 = consistent direction, -1.0 = flipping)")

                # Compute EWMA and EWSD if we have history
                if len(grad_norm_history) >= 3:
                    import numpy as np
                    recent = grad_norm_history[-10:]
                    ewma = np.mean(recent)
                    ewsd = np.std(recent)
                    print(f"  Grad EWMA: {ewma:.4f}  |  EWSD: {ewsd:.4f}")

            # Print additional metrics every 2 epochs
            if epoch % 2 == 0:
                print(f"  Grad noise (CV): {grad_noise_cv:.4f}  |  Dead ReLUs: {avg_dead_fraction:.2%}")

                # Per-layer step sizes
                if layer_steps:
                    print(f"  Layer steps: ", end="")
                    for layer, stats in layer_steps.items():
                        print(f"{layer}={stats['mean_step']:.6f} ", end="")
                    print()

                # Distance from init
                if dist_from_init:
                    print(f"  Distance from init: ", end="")
                    for layer, stats in dist_from_init.items():
                        print(f"{layer}={stats['mean_dist']:.3f} ", end="")
                    print()

            # Clear batch grad norms after each epoch
            batch_grad_norms = []

            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                model_name = "side_angles_best.pt" if not args.use_overhead else "multiview_best.pt"
                save_path = REPO / "indexes" / model_name
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model to {save_path}")

            # Learning rate decay
            if epoch % 5 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                    print(f"  → Learning rate: {param_group['lr']:.6f}")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("Training interrupted by user (Ctrl+C)")
        if best_val_loss < math.inf:
            print(f"Best validation MAE so far: {best_val_loss:.3f}")
            model_name = "side_angles_best.pt" if not args.use_overhead else "multiview_best.pt"
            print(f"Best model saved at: {REPO / 'indexes' / model_name}")
        print("=" * 70)
        return

    print("=" * 70)
    print(f"Training complete! Best validation MAE: {best_val_loss:.3f}")


if __name__ == "__main__":
    main()
