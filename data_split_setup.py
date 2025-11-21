# Create a small subset in your split files first
# e.g., only put 10-20 dish IDs in train.txt and test.txt

from src.data_loader import setup_data

# This will only download dishes listed in your split files
print("Starting data download...")
data_dir = setup_data(use_splits=True, force_download=True)
print(f"Data directory: {data_dir}")
