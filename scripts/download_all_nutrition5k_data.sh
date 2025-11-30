#!/bin/bash
# Download complete Nutrition5k dataset for multi-modal training
# Downloads: metadata, official splits, overhead RGB/depth, side angle frames
#
# Estimated total size: ~90 GB
# Estimated time: 2-3 hours on 100 Mbps connection

set -e

# Configuration
SOURCE_BUCKET="gs://nutrition5k_dataset/nutrition5k_dataset"
SIDE_FRAMES_BUCKET="gs://deepdiet-dataset"
LOCAL_DIR="$HOME/deepdiet/data/nutrition5k_dataset"

echo "=========================================="
echo "Nutrition5k Complete Dataset Download"
echo "=========================================="
echo ""
echo "This will download the COMPLETE Nutrition5k dataset:"
echo "  - Metadata files (dish nutritional labels)"
echo "  - Official train/test splits from paper"
echo "  - Overhead RGB images (~5 GB)"
echo "  - Overhead depth images (~5 GB)"
echo "  - Side angle frames (~80 GB)"
echo ""
echo "Total size: ~90 GB"
echo "Estimated time: 2-3 hours (depends on connection)"
echo ""
echo "Destination: $LOCAL_DIR"
echo ""

# Check available disk space
AVAILABLE=$(df -h "$HOME" | awk 'NR==2 {print $4}')
echo "Available disk space: $AVAILABLE"
echo ""

read -p "Continue with download? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Download cancelled"
    exit 1
fi

# Create local directory structure
mkdir -p "$LOCAL_DIR"
cd "$LOCAL_DIR"

echo ""
echo "=========================================="
echo "Step 1/5: Downloading Metadata"
echo "=========================================="
echo ""
echo "Downloading dish metadata CSVs (nutritional labels)..."

mkdir -p metadata
gsutil -m cp \
    "$SOURCE_BUCKET/metadata/dish_metadata_cafe1.csv" \
    "$SOURCE_BUCKET/metadata/dish_metadata_cafe2.csv" \
    metadata/

echo "✓ Metadata downloaded"
echo ""

echo "=========================================="
echo "Step 2/5: Downloading Official Splits"
echo "=========================================="
echo ""
echo "Downloading official train/test splits from paper..."

mkdir -p dish_ids/splits
gsutil -m rsync -r "$SOURCE_BUCKET/dish_ids/splits/" dish_ids/splits/
gsutil cp "$SOURCE_BUCKET/dish_ids/README" dish_ids/

echo "✓ Splits downloaded"
echo ""

echo "=========================================="
echo "Step 3/5: Downloading Overhead RGB Images"
echo "=========================================="
echo ""
echo "Downloading overhead RGB images (~5 GB)..."
echo "This will take 5-10 minutes..."
echo ""

mkdir -p imagery/realsense_overhead

# Download all rgb.png files
# Using rsync with pattern matching for rgb.png only
gsutil -m rsync -r \
    -x '.*depth_color\.png$' \
    "$SOURCE_BUCKET/imagery/realsense_overhead/" \
    imagery/realsense_overhead/

echo "✓ Overhead RGB images downloaded"
echo ""

echo "=========================================="
echo "Step 4/5: Downloading Overhead Depth Images"
echo "=========================================="
echo ""
echo "Downloading overhead depth images (~5 GB)..."
echo "This will take 5-10 minutes..."
echo ""

# Depth files are already included in the rsync above
# (we only excluded depth_color.png which is optional)

echo "✓ Overhead depth images downloaded"
echo ""

echo "=========================================="
echo "Step 5/5: Downloading Side Angle Frames"
echo "=========================================="
echo ""
echo "Downloading side angle frames from your bucket (~80 GB)..."
echo "Source: $SIDE_FRAMES_BUCKET"
echo "This is the largest component and will take 1-2 hours..."
echo ""
echo "Progress will be shown by gsutil..."
echo ""

mkdir -p imagery/side_angles

# Download all side angle frames from your bucket
# Exclude .h264 video files (we only need the sampled JPEG frames)
echo "Downloading from deepdiet-dataset bucket (extracted frames)..."
gsutil -m rsync -r \
    -x '.*\.h264$' \
    "$SIDE_FRAMES_BUCKET/nutrition5k_dataset/imagery/side_angles/" \
    imagery/side_angles/

echo "✓ Side angle frames downloaded"
echo ""

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""

# Verify downloads
echo "Verifying downloads..."
echo ""
echo "Metadata files:"
ls -lh metadata/*.csv 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

echo "Official splits:"
ls -1 dish_ids/splits/*.txt 2>/dev/null | wc -l | xargs echo "  " "split files"
echo ""

echo "Overhead images:"
find imagery/realsense_overhead -name "rgb.png" 2>/dev/null | wc -l | xargs echo "  " "RGB images"
find imagery/realsense_overhead -name "depth_raw.png" 2>/dev/null | wc -l | xargs echo "  " "depth images"
echo ""

echo "Side angle frames:"
find imagery/side_angles -name "*.jpeg" 2>/dev/null | wc -l | xargs echo "  " "JPEG frames"
echo ""

echo "Total storage used:"
du -sh .
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Generate training CSVs:"
echo ""
echo "   python scripts/generate_training_csvs.py"
echo ""
echo "2. Start training:"
echo ""
echo "   # Overhead RGB + depth only"
echo "   python src/train.py --use-overhead --use-depth --epochs 50"
echo ""
echo "   # Full multi-view (side + overhead + depth)"
echo "   python src/train.py --use-overhead --use-depth --epochs 50"
echo ""
echo "Dataset ready!"
echo ""
