from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_NORM = T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

class OverheadRGBOnly(Dataset):
    def __init__(self, csv_path, data_root="data/nutrition5k_dataset"):
        self.df = pd.read_csv(csv_path)
        self.root = Path(data_root)
        self.tf = T.Compose([T.Resize((256,256)), T.ToTensor(), IMAGENET_NORM])
        self.tkeys = ["cal","mass","fat","carb","protein"]

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        x = self.tf(Image.open(self.root/row["rgb"]).convert("RGB"))
        y = torch.tensor([row[k] for k in self.tkeys], dtype=torch.float32)
        return x, y