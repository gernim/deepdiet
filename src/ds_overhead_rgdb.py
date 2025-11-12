# src/ds_overhead_rgbd.py
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

IMAGENET_NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def rgb_tf():
    return T.Compose([T.Resize((256,256)), T.ToTensor(), IMAGENET_NORM])

def depth_tf():
    # resize to match RGB; keep as 1 channel float meters, clipped at 0.4m per dataset doc
    return T.Compose([T.Resize((256,256), interpolation=T.InterpolationMode.NEAREST)])

class OverheadRGBDDataset(Dataset):
    def __init__(self, csv_path, data_root="data/nutrition5k_dataset"):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.rgb_t = rgb_tf()
        self.depth_t = depth_tf()
        self.targets = ["cal","mass","fat","carb","protein"]

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        # RGB
        rgb = Image.open(self.root/row["rgb"]).convert("RGB")
        x_rgb = self.rgb_t(rgb)  # [3,H,W]

        # Depth raw uint16 → meters, clip to 0.4m
        depth_raw = Image.open(self.root/row["depth_raw"])
        d_np = np.array(depth_raw).astype(np.float32) / 10000.0
        d_np = np.clip(d_np, 0.0, 0.4)
        d_img = Image.fromarray((d_np*65535.0/0.4).astype(np.uint16))
        d_img = self.depth_t(d_img)
        d = torch.from_numpy(d_np).unsqueeze(0)  # original meters (1,H,W)
        d = T.functional.resize(d, [256,256], antialias=False)

        # Fuse as 4th channel (simple baseline)
        x = torch.cat([x_rgb, d], dim=0)  # [4,256,256]

        y = torch.tensor([row[t] for t in self.targets], dtype=torch.float32)
        return x, y