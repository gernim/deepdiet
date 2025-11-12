# src/ds_side_frames.py
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

IMAGENET_NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def default_rgb_tf(center_crop=800):
    return T.Compose([
        T.CenterCrop(center_crop),   # plate centered in frame
        T.Resize((256,256)),
        T.ToTensor(),
        IMAGENET_NORM,
    ])

class SideFramesDataset(Dataset):
    def __init__(self, csv_path, data_root="data/nutrition5k_dataset", transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.tf = transform or default_rgb_tf()
        self.targets = ["cal","mass","fat","carb","protein"]

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root/row["image"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor([row[t] for t in self.targets], dtype=torch.float32)
        return x, y