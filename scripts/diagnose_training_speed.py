#!/usr/bin/env python3
"""
Quick diagnostic to identify training bottlenecks.
Run this to test data loading, forward pass, and backward pass speeds.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import time
from src.train import MultiViewDataset, MultiViewModel, get_device, REPO

def diagnose():
    device = get_device()
    print(f"Device: {device}")
    print("=" * 70)

    # Test 1: Data loading speed (GCS)
    print("\n[Test 1] Data Loading Speed (GCS)...")
    train_csv = REPO / "indexes" / "side_frames_train.csv"

    dataset = MultiViewDataset(
        overhead_csv=None,
        side_csv=train_csv,
        data_root=REPO / "data" / "nutrition5k_dataset",
        train=True,
        max_side_frames=16,
        use_overhead=False,
        use_gcs=True,
        gcs_bucket='deepdiet-dataset',
        gcs_prefix='nutrition5k_dataset/'
    )

    print(f"Dataset size: {len(dataset)} dishes")

    # Time loading 10 samples
    start = time.time()
    for i in range(10):
        sample = dataset[i]
    elapsed = time.time() - start
    print(f"✓ Loaded 10 samples in {elapsed:.2f}s ({elapsed/10:.2f}s per sample)")
    if elapsed/10 > 5:
        print("⚠️  WARNING: Data loading is VERY slow (>5s per sample)!")
        print("   This is likely the bottleneck. GCS streaming may be too slow.")

    # Test 2: Model forward pass
    print("\n[Test 2] Model Forward Pass Speed...")
    model = MultiViewModel(
        num_side_frames=16,
        use_overhead=False,
        use_depth=True,
        chunk_size=4
    ).to(device)

    # Create dummy batch
    batch_size = 8
    side_frames = torch.randn(batch_size, 16, 3, 256, 256).to(device)

    # Warmup
    with torch.no_grad():
        _ = model(side_frames)

    # Time forward pass
    start = time.time()
    for _ in range(10):
        with torch.no_grad():
            pred = model(side_frames)
    elapsed = time.time() - start
    print(f"✓ 10 forward passes in {elapsed:.2f}s ({elapsed/10:.2f}s per batch)")
    if elapsed/10 > 2:
        print("⚠️  WARNING: Forward pass is slow (>2s per batch)")

    # Test 3: Backward pass
    print("\n[Test 3] Backward Pass Speed...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.L1Loss()

    targets = torch.randn(batch_size, 5).to(device)

    start = time.time()
    for _ in range(10):
        optimizer.zero_grad()
        pred = model(side_frames)
        loss = criterion(pred, targets)
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    print(f"✓ 10 full iterations in {elapsed:.2f}s ({elapsed/10:.2f}s per iteration)")

    # Test 4: DataLoader with workers
    print("\n[Test 4] DataLoader with Batching...")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,  # GCS requires 0 workers
        pin_memory=(device.type == 'cuda')
    )

    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= 5:  # Test first 5 batches
            break
        _ = batch['side_frames'].to(device)
    elapsed = time.time() - start
    print(f"✓ Loaded 5 batches in {elapsed:.2f}s ({elapsed/5:.2f}s per batch)")

    if elapsed/5 > 10:
        print("⚠️  CRITICAL: DataLoader is extremely slow (>10s per batch)!")
        print("   Problem: GCS streaming is the bottleneck")
        print("   Solutions:")
        print("   1. Download dataset locally instead of streaming from GCS")
        print("   2. Use a persistent disk attached to the VM")
        print("   3. Reduce max_frames from 16 to 8")

    # Calculate expected epoch time
    print("\n" + "=" * 70)
    print("PROJECTION:")
    batches_per_epoch = len(dataset) // 8
    time_per_batch = elapsed / 5
    estimated_epoch_time = batches_per_epoch * time_per_batch
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Time per batch: {time_per_batch:.2f}s")
    print(f"  Estimated epoch time: {estimated_epoch_time/3600:.2f} hours")
    print("=" * 70)

if __name__ == "__main__":
    diagnose()
