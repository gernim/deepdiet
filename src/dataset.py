import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Nutrition5KDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        row = self.df.iloc[idx, 1]

        return img, row