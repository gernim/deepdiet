from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

TARGETS = ["cal", "mass", "fat", "carb", "protein"]

class MultiViewDataset(Dataset):
    def __init__(self, split_file, data_root, train=True, max_side_frames=16,
                 use_side_frames=True, use_overhead=False, use_depth=False, image_size=256):
        self.data_root = Path(data_root)
        self.train = train
        self.max_side_frames = max_side_frames
        self.use_side_frames = use_side_frames
        self.use_overhead = use_overhead
        self.use_depth = use_depth
        self.image_size = image_size

        split = pd.read_csv(split_file)
        all_dish_ids = split['dish_id'].tolist()

        cafe1_path = self.data_root / "metadata" / "dish_metadata_cafe1.csv"
        cafe2_path = self.data_root / "metadata" / "dish_metadata_cafe2.csv"

        # Read metadata robustly (CSVs may have irregular rows and column names)
        meta_dfs = []
        for path in [cafe1_path, cafe2_path]:
            if path.exists():
                # Try reading with headers first
                df = pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)

                # Build case-insensitive column mapping
                colmap = {c.lower(): c for c in df.columns if isinstance(c, str)}

                # Find columns with flexible matching
                def pick(*cands):
                    for cand in cands:
                        key = str(cand).lower()
                        if key in colmap:
                            return colmap[key]
                    return None

                col_dish = pick("dish_id", "id", "dish")
                col_cal = pick("total_calories", "calories_total", "total_kcal", "kcal")
                col_mass = pick("total_mass", "mass_total", "mass_g", "total_mass_g")
                col_fat = pick("total_fat", "fat_total", "fat_g")
                col_carb = pick("total_carb", "carb_total", "carbs_total", "carb_g", "carbs_g")
                col_prot = pick("total_protein", "protein_total", "protein_g")

                # If headers found, use them
                if all(c is not None for c in [col_dish, col_cal, col_mass, col_fat, col_carb, col_prot]):
                    df = df[[col_dish, col_cal, col_mass, col_fat, col_carb, col_prot]].rename(columns={
                        col_dish: "dish_id",
                        col_cal: "cal",
                        col_mass: "mass",
                        col_fat: "fat",
                        col_carb: "carb",
                        col_prot: "protein",
                    })
                else:
                    # Fallback: assume headerless CSV with standard column order
                    df = pd.read_csv(path, header=None, engine="python", on_bad_lines="skip")
                    df = df.iloc[:, :6]  # Take first 6 columns
                    df.columns = ["dish_id", "cal", "mass", "fat", "carb", "protein"]

                meta_dfs.append(df)

        metadata = pd.concat(meta_dfs, ignore_index=True)

        # Convert nutritional columns to numeric
        for col in TARGETS:
            metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

        # Drop rows with missing values
        metadata = metadata.dropna(subset=["dish_id"] + TARGETS).set_index("dish_id")
        self.metadata = metadata

        valid_dish_ids = []

        for dish_id in all_dish_ids:
            if dish_id not in self.metadata.index:
                continue

            is_valid = True

            if self.use_side_frames:
                side_dir = self.data_root / "imagery" / "side_angles" / dish_id / "frames_sampled5"
                if not side_dir.is_dir() or not list(side_dir.glob("*.jpeg")):
                    is_valid = False

            if self.use_overhead:
                overhead_path = self.data_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
                if not overhead_path.is_file() or overhead_path.stat().st_size == 0:
                    is_valid = False

            if self.use_depth:
                depth_path = self.data_root / "imagery" / "realsense_overhead" / dish_id / "depth_raw.png"
                if not depth_path.is_file() or depth_path.stat().st_size == 0:
                    is_valid = False

            if is_valid:
                valid_dish_ids.append(dish_id)

        self.dish_ids = valid_dish_ids
        print(f"Loaded {len(self.dish_ids)} dishes from dataset.")

        # Calculate resize dimensions (slightly larger for cropping)
        resize_size = int(image_size * 1.125)  # 12.5% larger for crop

        if train:
            self.side_transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(image_size),
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

        self.overhead_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.depth_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

    def _load_side_frames(self, dish_id):
        side_dir = self.data_root / "imagery" / "side_angles" / dish_id / "frames_sampled5"
        all_frames = sorted(list(side_dir.glob("*.jpeg")))

        # Group frames by camera
        from collections import defaultdict
        camera_frames = defaultdict(list)
        for frame_path in all_frames:
            camera = frame_path.stem.split('_')[1]
            camera_frames[camera].append(frame_path)

        # Sort frames within each camera
        for camera in camera_frames:
            camera_frames[camera].sort()

        cameras = sorted(camera_frames.keys())
        frames_per_camera = self.max_side_frames // len(cameras)

        # Sample evenly from each camera
        sampled_per_camera = {}
        for camera in cameras:
            available = camera_frames[camera]
            n = len(available)
            if n == 0:
                continue
            if frames_per_camera >= n:
                sampled_per_camera[camera] = available
            else:
                indices = [int(i * n / frames_per_camera) for i in range(frames_per_camera)]
                sampled_per_camera[camera] = [available[i] for i in indices]

        # Interleave cameras: [A0, B0, C0, D0, A1, B1, C1, D1, ...]
        selected_frames = []
        max_len = max(len(v) for v in sampled_per_camera.values())
        for i in range(max_len):
            for camera in cameras:
                if i < len(sampled_per_camera.get(camera, [])):
                    selected_frames.append(sampled_per_camera[camera][i])

        return selected_frames[:self.max_side_frames]

    def __len__(self):
        return len(self.dish_ids)

    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        row = self.metadata.loc[dish_id]
        targets = torch.tensor([row[t] for t in TARGETS], dtype=torch.float32)

        if self.use_overhead:
            overhead_path = self.data_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
            overhead_img = Image.open(overhead_path).convert("RGB")
            overhead_img = self.overhead_transform(overhead_img)
        else:
            overhead_img = None

        if self.use_depth:
            depth_path = self.data_root / "imagery" / "realsense_overhead" / dish_id / "depth_raw.png"
            depth_img = Image.open(depth_path).convert("L")
            depth_img = self.depth_transform(depth_img)
        else:
            depth_img = None

        if self.use_side_frames:
            frame_paths = self._load_side_frames(dish_id)
            side_frames = []

            for frame_path in frame_paths:
                try:
                    frame = Image.open(frame_path).convert('RGB')
                    frame = self.side_transform(frame)
                    side_frames.append(frame)
                except Exception as e:
                    print(f"Error loading frame {frame_path}: {e}")
                    continue

            while len(side_frames) < self.max_side_frames:
                side_frames.append(torch.zeros(3, self.image_size, self.image_size))
            side_frames = torch.stack(side_frames)
        else:
            side_frames = None

        result = {
            'targets': targets,
            'dish_id': dish_id
        }

        if side_frames is not None:
            result['side_frames'] = side_frames

        if overhead_img is not None:
            result['overhead_rgb'] = overhead_img

        if depth_img is not None:
            result['overhead_depth'] = depth_img

        return result