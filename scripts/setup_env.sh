#!/bin/bash
set -e  # Exit on error

echo "🚀 Setting up deepdiet environment..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create environment from environment.yml
echo "📦 Creating conda environment from environment.yml..."
conda env create -f environment.yml

echo "✅ Environment created!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate deepdiet"
echo ""
echo "To test the installation, run:"
echo "  conda activate deepdiet"
echo "  python scripts/test_imports.py"ed on your list)