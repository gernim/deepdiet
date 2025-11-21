#!/usr/bin/env python3
"""Test that all required packages can be imported."""
import sys


def test_imports():
    all_good = True

    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "torchvision"),
        ("google.cloud.storage", "google-cloud-storage"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("PIL", "Pillow"),
        ("cv2", "opencv-python"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
    ]

    print("Testing imports...\n")

    for module_name, package_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package_name:25s} {version}")
        except ImportError as e:
            print(f"✗ {package_name:25s} NOT FOUND")
            all_good = False

    # Special checks
    try:
        import torch
        print(f"\n  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU devices: {torch.cuda.device_count()}")
    except:
        pass

    print("\n" + "=" * 50)
    if all_good:
        print("✅ All imports successful!")
        return 0
    else:
        print("❌ Some imports failed. Please install missing packages.")
        return 1


if __name__ == "__main__":
    sys.exit(test_imports())