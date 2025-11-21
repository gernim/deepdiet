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

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "nutrition5k_dataset"
OVERHEAD_TRAIN_CSV = REPO / "indexes" / "overhead_train.csv"
OVERHEAD_TEST_CSV = REPO / "indexes" / "overhead_test.csv"
SIDE_TRAIN_CSV = REPO / "indexes" / "side_frames_train.csv"
SIDE_TEST_CSV = REPO / "indexes" / "side_frames_test.csv"
TARGETS = ["cal", "mass", "fat", "carb", "protein"]

class MultiViewDataset(Dataset):
    """Dataset that uses side angle views, optionally with overhead RGB/depth."""

    def __init__(self, overhead_csv, side_csv, data_root, train=True, max_side_frames=16, use_overhead=False):
        """
        Args:
            overhead_csv: Path to overhead images CSV (can be None if use_overhead=False)
            side_csv: Path to side angle frames CSV
            data_root: Root directory for data
            train: Training mode (enables augmentation)
            max_side_frames: Maximum number of side frames to use per dish
            use_overhead: Whether to load overhead RGB/depth images
        """
        self.data_root = Path(data_root)
        self.train = train
        self.max_side_frames = max_side_frames
        self.use_overhead = use_overhead

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

        # Transforms for RGB images
        if train:
            self.rgb_transform = transforms.Compose([
                transforms.Resize(288),
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize(288),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])

        # Depth transform (normalize to 0-1)
        self.depth_transform = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256) if not train else transforms.RandomCrop(256),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']

        # Load overhead RGB and depth if using overhead
        if self.use_overhead:
            overhead_rgb_path = self.data_root / row['rgb']
            overhead_rgb = Image.open(overhead_rgb_path).convert('RGB')
            overhead_rgb = self.rgb_transform(overhead_rgb)

            # Load overhead depth (if available)
            overhead_depth_path = self.data_root / row['depth_raw']
            try:
                overhead_depth = Image.open(overhead_depth_path).convert('L')
                overhead_depth = self.depth_transform(overhead_depth)
            except:
                # If depth not available, use zeros
                overhead_depth = torch.zeros(1, 256, 256)
        else:
            overhead_rgb = None
            overhead_depth = None

        # Load side angle frames
        side_frame_paths = self.side_frames[dish_id][:self.max_side_frames]
        side_frames = []
        for frame_path in side_frame_paths:
            try:
                frame = Image.open(self.data_root / frame_path).convert('RGB')
                frame = self.rgb_transform(frame)
                side_frames.append(frame)
            except:
                pass

        # Pad or truncate to max_side_frames
        while len(side_frames) < self.max_side_frames:
            side_frames.append(torch.zeros(3, 256, 256))
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
    """Multi-view model using side angles, optionally with overhead RGB/depth."""

    def __init__(self, num_side_frames=16, use_overhead=False, use_depth=True):
        super().__init__()
        self.use_overhead = use_overhead
        self.use_depth = use_depth and use_overhead  # Depth only makes sense with overhead

        # Side angle encoder (shared across frames)
        self.side_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        side_feat_dim = self.side_encoder.fc.in_features
        self.side_encoder.fc = nn.Identity()

        # Aggregation for side frames
        self.side_aggregator = nn.LSTM(side_feat_dim, 256, batch_first=True, bidirectional=True)

        total_feat_dim = 512  # 512 from bidirectional LSTM

        # Overhead encoders (if using overhead)
        if use_overhead:
            # Overhead RGB encoder
            self.overhead_rgb_encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            overhead_feat_dim = self.overhead_rgb_encoder.fc.in_features
            self.overhead_rgb_encoder.fc = nn.Identity()
            total_feat_dim += overhead_feat_dim

            # Overhead depth encoder (if using depth)
            if self.use_depth:
                self.overhead_depth_encoder = models.resnet18(weights=None)
                # Modify first conv for single channel
                self.overhead_depth_encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                depth_feat_dim = self.overhead_depth_encoder.fc.in_features
                self.overhead_depth_encoder.fc = nn.Identity()
                total_feat_dim += depth_feat_dim

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(total_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(TARGETS))
        )

    def forward(self, side_frames, overhead_rgb=None, overhead_depth=None):
        batch_size = side_frames.size(0)

        # Encode side frames
        num_frames = side_frames.size(1)
        side_frames_flat = side_frames.view(batch_size * num_frames, 3, 256, 256)
        side_feat = self.side_encoder(side_frames_flat)
        side_feat = side_feat.view(batch_size, num_frames, -1)

        # Aggregate side features with LSTM
        side_feat, _ = self.side_aggregator(side_feat)
        side_feat = side_feat[:, -1, :]  # Take last hidden state

        # Fuse features
        features = [side_feat]

        if self.use_overhead and overhead_rgb is not None:
            overhead_rgb_feat = self.overhead_rgb_encoder(overhead_rgb)
            features.append(overhead_rgb_feat)

            if self.use_depth and overhead_depth is not None:
                overhead_depth_feat = self.overhead_depth_encoder(overhead_depth)
                features.append(overhead_depth_feat)

        combined = torch.cat(features, dim=1)

        # Predict
        out = self.fusion(combined)
        return out


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mae_metric(pred, target):
    return (pred - target).abs().mean().item()


def main():
    parser = argparse.ArgumentParser(description='Train DeepDiet multi-view model')
    parser.add_argument('--use-overhead', action='store_true',
                        help='Use overhead RGB and depth images (default: side angles only)')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable depth images when using overhead (only works with --use-overhead)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train (default: 20)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--max-frames', type=int, default=16,
                        help='Maximum number of side angle frames per dish (default: 16)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"Mode: {'Side angles + overhead RGB/depth' if args.use_overhead else 'Side angles only'}")
    if args.use_overhead and args.no_depth:
        print("  (depth disabled)")

    # Create datasets
    print("\nLoading datasets...")
    train_ds = MultiViewDataset(
        OVERHEAD_TRAIN_CSV if args.use_overhead else None,
        SIDE_TRAIN_CSV,
        DATA_ROOT,
        train=True,
        max_side_frames=args.max_frames,
        use_overhead=args.use_overhead
    )
    val_ds = MultiViewDataset(
        OVERHEAD_TEST_CSV if args.use_overhead else None,
        SIDE_TEST_CSV,
        DATA_ROOT,
        train=False,
        max_side_frames=args.max_frames,
        use_overhead=args.use_overhead
    )

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=False)

    # Create model
    print("\nInitializing model...")
    model = MultiViewModel(
        num_side_frames=args.max_frames,
        use_overhead=args.use_overhead,
        use_depth=not args.no_depth
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Training loop
    best_val_loss = math.inf

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss, train_mae = 0.0, 0.0

        for batch in train_dl:
            side_frames = batch['side_frames'].to(device)
            targets = batch['targets'].to(device)

            overhead_rgb = batch.get('overhead_rgb')
            overhead_depth = batch.get('overhead_depth')

            if overhead_rgb is not None:
                overhead_rgb = overhead_rgb.to(device)
            if overhead_depth is not None:
                overhead_depth = overhead_depth.to(device)

            optimizer.zero_grad()
            pred = model(side_frames, overhead_rgb, overhead_depth)
            loss = criterion(pred, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * side_frames.size(0)
            train_mae += mae_metric(pred.detach(), targets.detach()) * side_frames.size(0)

        train_loss /= len(train_ds)
        train_mae /= len(train_ds)

        # Validate
        model.eval()
        val_loss, val_mae = 0.0, 0.0

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

                pred = model(side_frames, overhead_rgb, overhead_depth)
                loss = criterion(pred, targets)

                val_loss += loss.item() * side_frames.size(0)
                val_mae += mae_metric(pred, targets) * side_frames.size(0)

        val_loss /= len(val_ds)
        val_mae /= len(val_ds)

        print(f"[Epoch {epoch:02d}/{args.epochs}] "
              f"Train MSE: {train_loss:.3f} MAE: {train_mae:.3f} | "
              f"Val MSE: {val_loss:.3f} MAE: {val_mae:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_name = "side_angles_best.pt" if not args.use_overhead else "multiview_best.pt"
            save_path = REPO / "indexes" / model_name
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model to {save_path}")

        # Learning rate decay
        if epoch % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"  → Learning rate: {param_group['lr']:.6f}")

    print("=" * 70)
    print(f"Training complete! Best validation MSE: {best_val_loss:.3f}")


if __name__ == "__main__":
    main()
