#!/usr/bin/env python3
"""
Selective download script - downloads ONLY the side frame JPEGs needed for training.
Skips videos (.h264), depth maps, and overhead images.
"""
import pandas as pd
from pathlib import Path
from google.cloud import storage
from tqdm import tqdm
import sys

# Configuration
BUCKET_NAME = "deepdiet-dataset"
BUCKET_PREFIX = "nutrition5k_dataset/"
LOCAL_DIR = Path.home() / "deepdiet" / "data" / "nutrition5k_dataset"
TRAIN_CSV = Path("indexes/side_frames_train.csv")
TEST_CSV = Path("indexes/side_frames_test.csv")

def get_unique_files():
    """Extract unique file paths from CSVs."""
    files = set()

    for csv_file in [TRAIN_CSV, TEST_CSV]:
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            files.update(df['image'].unique())
        else:
            print(f"Warning: {csv_file} not found")

    return sorted(files)

def estimate_size(num_files):
    """Estimate download size (rough estimate)."""
    avg_size_kb = 500  # Average JPEG size
    total_mb = (num_files * avg_size_kb) / 1024
    return total_mb

def download_files(files):
    """Download files from GCS."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    print(f"\nDownloading {len(files)} files...")
    print(f"Destination: {LOCAL_DIR}")
    print()

    for file_path in tqdm(files, desc="Downloading"):
        local_path = LOCAL_DIR / file_path

        # Skip if already exists
        if local_path.exists():
            skipped += 1
            continue

        # Create parent directories
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Download from GCS
            blob_path = BUCKET_PREFIX + file_path
            blob = bucket.blob(blob_path)
            blob.download_to_filename(str(local_path))
            downloaded += 1
        except Exception as e:
            print(f"\nFailed to download {file_path}: {e}")
            failed += 1

    return downloaded, skipped, failed

def main():
    print("=" * 70)
    print("Selective Download - Side Frames Only")
    print("=" * 70)
    print()

    # Get file list
    print("Step 1: Extracting file paths from CSVs...")
    files = get_unique_files()

    if not files:
        print("Error: No files found in CSVs!")
        sys.exit(1)

    print(f"Found {len(files)} unique files")

    # Estimate size
    estimated_mb = estimate_size(len(files))
    estimated_gb = estimated_mb / 1024
    print(f"Estimated download size: ~{estimated_mb:.0f} MB (~{estimated_gb:.1f} GB)")
    print()

    # Check if directory exists
    if LOCAL_DIR.exists():
        existing = list(LOCAL_DIR.glob("**/*.jpeg"))
        print(f"Found {len(existing)} existing files (will skip)")
        print()

    # Confirm
    response = input("Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        sys.exit(0)

    # Download
    print()
    print("Step 2: Downloading files...")
    print("This may take 30-60 minutes depending on network speed")

    try:
        downloaded, skipped, failed = download_files(files)

        print()
        print("=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print()
        print(f"Downloaded: {downloaded} files")
        print(f"Skipped (already exist): {skipped} files")
        if failed > 0:
            print(f"Failed: {failed} files")
        print()
        print(f"Files saved to: {LOCAL_DIR}")
        print()
        print("Next steps:")
        print("Train WITHOUT --use-gcs flag:")
        print()
        print("  python src/train.py \\")
        print("    --train-csv indexes/side_frames_train.csv \\")
        print("    --test-csv indexes/side_frames_test.csv \\")
        print("    --epochs 50 \\")
        print("    --batch-size 8 \\")
        print("    --chunk-size 4")
        print()

    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
