import os
from google.cloud import storage
from pathlib import Path


class DataPathManager:
    """Manages data paths for both local and GCS environments."""

    def __init__(self,
                 bucket_name='deepdiet-dataset',
                 gcs_prefix='nutrition5k_dataset/',
                 local_data_dir='./data/nutrition5k_dataset',
                 project_id='cs230-project-478801',
                 splits_dir='./data/nutrition5k_dataset/dish_ids/splits'):

        self.bucket_name = bucket_name
        self.gcs_prefix = gcs_prefix
        self.local_data_dir = local_data_dir
        self.project_id = project_id
        self.splits_dir = splits_dir

        # Detect if running on GCP (common env variable)
        self.is_gcp = os.getenv('GOOGLE_CLOUD_PROJECT') is not None or \
                      os.getenv('GCP_PROJECT') is not None

    def read_dish_ids_from_splits(self):
        """Read dish IDs from split files."""
        splits_path = Path(self.splits_dir)
        
        # First, ensure we have the splits files locally
        if not splits_path.exists():
            print("Splits directory not found, downloading split files from GCS...")
            self._download_splits()
        
        dish_ids = set()
        
        # Read all split files to get the dish IDs we need
        split_files = ['train.txt', 'test.txt']  # Add more as needed
        
        for split_file in split_files:
            split_path = splits_path / split_file
            if split_path.exists():
                with open(split_path, 'r') as f:
                    ids = [line.strip() for line in f if line.strip()]
                    dish_ids.update(ids)
                    print(f"Loaded {len(ids)} dish IDs from {split_file}")
            else:
                print(f"Warning: {split_file} not found at {split_path}")
        
        print(f"Total unique dish IDs to download: {len(dish_ids)}")
        return dish_ids

    def _download_splits(self):
        """Download just the split files from GCS."""
        os.makedirs(self.splits_dir, exist_ok=True)
        
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        # Download the dish_ids directory structure
        prefix = self.gcs_prefix + 'dish_ids/'
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            
            relative_path = blob.name[len(self.gcs_prefix):]
            local_path = os.path.join(self.local_data_dir, relative_path)
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_path)

    def _get_dish_file_paths(self, dish_id):
        """Generate all file paths for a given dish ID based on Nutrition5k structure."""
        files_to_download = []

        # Based on the Nutrition5k dataset structure:
        # imagery/realsense_overhead/dish_<dish_id>/
        #   - rgb.png
        #   - depth_color.png
        #   - depth_raw.png
        # imagery/realsense_top_down/dish_<dish_id>/
        #   - rgb.png
        #   - depth_color.png
        #   - depth_raw.png
        # imagery/side_angles/dish_<dish_id>/frames_sampled5/
        #   - frame_000.png, frame_001.png, ... (typically ~16 frames per dish)

        # Handle dish_id that may or may not already have "dish_" prefix
        dish_folder = dish_id if dish_id.startswith('dish_') else f"dish_{dish_id}"
        
        # Overhead imagery
        overhead_paths = [
            f"imagery/realsense_overhead/{dish_folder}/rgb.png",
            f"imagery/realsense_overhead/{dish_folder}/depth_color.png",
            f"imagery/realsense_overhead/{dish_folder}/depth_raw.png",
        ]

        files_to_download.extend(overhead_paths)
        
        # Side angle sampled frames (every 5th frame from video)
        # Note: Number of frames varies per dish, we'll discover them dynamically
        side_angles_prefix = f"imagery/side_angles/{dish_folder}/frames_sampled5/"
        files_to_download.append(side_angles_prefix)  # This will be handled specially
        
        return files_to_download

    def download_data_if_needed(self, use_splits=True):
        """Download data from GCS if not already present locally.
        
        Args:
            use_splits: If True, only download dishes listed in split files.
                       If False, download all metadata and imagery.
        """
        
        # Always download metadata and dish_ids files (they're small)
        print("Downloading metadata and dish_ids...")
        self._download_metadata_and_dish_ids()
        
        if not use_splits:
            # Download everything
            print("Downloading all imagery (this may take a while)...")
            self._download_all_imagery()
        else:
            # Download only dishes in splits
            dish_ids = self.read_dish_ids_from_splits()
            self._download_dishes_by_id(dish_ids)
        
        print(f"Download complete! Data saved to {self.local_data_dir}")
        return self.local_data_dir

    def _download_metadata_and_dish_ids(self):
        """Download metadata and dish_ids directories (small files)."""
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        # Download metadata and dish_ids
        for prefix in ['metadata/', 'dish_ids/']:
            full_prefix = self.gcs_prefix + prefix
            blobs = bucket.list_blobs(prefix=full_prefix)
            
            for blob in blobs:
                if blob.name.endswith('/'):
                    continue
                
                relative_path = blob.name[len(self.gcs_prefix):]
                local_path = os.path.join(self.local_data_dir, relative_path)
                
                # Skip if already exists
                if os.path.exists(local_path):
                    continue
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                print(f"Downloading {blob.name}...")
                blob.download_to_filename(local_path)

    def _download_all_imagery(self):
        """Download all imagery from GCS."""
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        prefix = self.gcs_prefix + 'imagery/'
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            
            relative_path = blob.name[len(self.gcs_prefix):]
            local_path = os.path.join(self.local_data_dir, relative_path)
            
            # Skip if already exists
            if os.path.exists(local_path):
                continue
            
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            print(f"Downloading {blob.name}...")
            blob.download_to_filename(local_path)

    def _download_dishes_by_id(self, dish_ids):
        """Download only specific dishes by their IDs."""
        storage_client = storage.Client(project=self.project_id)
        bucket = storage_client.bucket(self.bucket_name)
        
        total_dishes = len(dish_ids)
        for idx, dish_id in enumerate(dish_ids, 1):
            print(f"Downloading dish {idx}/{total_dishes}: {dish_id}")
            
            # Get all file paths for this dish
            file_paths = self._get_dish_file_paths(dish_id)
            
            for file_path in file_paths:
                gcs_path = self.gcs_prefix + file_path
                local_path = os.path.join(self.local_data_dir, file_path)
                
                # Special handling for side_angles directory (multiple frames)
                if file_path.endswith('frames_sampled5/'):
                    self._download_side_angle_frames(bucket, dish_id, gcs_path, local_path)
                    continue
                
                # Skip if already exists
                if os.path.exists(local_path):
                    continue
                
                # Try to download (some files might not exist for all dishes)
                try:
                    blob = bucket.blob(gcs_path)
                    if blob.exists():
                        os.makedirs(os.path.dirname(local_path), exist_ok=True)
                        blob.download_to_filename(local_path)
                    else:
                        print(f"  Skipping {file_path} (not found in GCS)")
                except Exception as e:
                    print(f"  Error downloading {file_path}: {e}")

    def _download_side_angle_frames(self, bucket, dish_id, gcs_prefix, local_dir):
        """Download all sampled frames for a dish's side angles."""
        os.makedirs(local_dir, exist_ok=True)
        
        # List all frames in the frames_sampled5 directory
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        frame_count = 0
        for blob in blobs:
            if blob.name.endswith('/'):
                continue
            
            # Extract filename (e.g., frame_000.png)
            filename = blob.name.split('/')[-1]
            local_path = os.path.join(local_dir, filename)
            
            # Skip if already exists
            if os.path.exists(local_path):
                continue
            
            try:
                blob.download_to_filename(local_path)
                frame_count += 1
            except Exception as e:
                print(f"  Error downloading frame {filename}: {e}")
        
        if frame_count > 0:
            print(f"  Downloaded {frame_count} side angle frames for dish_{dish_id}")

    def get_data_path(self):
        """Get the appropriate data path."""
        return self.local_data_dir


def setup_data(force_download=False, use_splits=True):
    """Convenience function to set up data paths.

    Args:
        force_download: Force re-download even if data exists
        use_splits: Only download dishes in split files (recommended for local testing)
    """
    manager = DataPathManager()

    if force_download or not os.path.exists(manager.local_data_dir):
        manager.download_data_if_needed(use_splits=use_splits)

    return manager.get_data_path()