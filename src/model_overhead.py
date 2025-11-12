# src/train_min_overhead.py
from pathlib import Path
import os, math, csv
import pandas as pd
from PIL import Image
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "nutrition5k_dataset"
TRAIN_CSV = REPO / "indexes" / "overhead_train.csv"
VAL_CSV   = REPO / "indexes" / "overhead_test.csv"

# Targets: order we’ll use everywhere
TARGETS = ["cal","mass","fat","carb","protein"]

class OverheadDataset(Dataset):
    def __init__(self, index_csv: Path, data_root: Path, train=True):
        df = pd.read_csv(index_csv)
        # minimal columns required
        # dish_id,rgb,depth_raw,cal,mass,fat,carb,protein
        # depth_raw is ignored in this simple baseline
        keep = ["dish_id","rgb"] + TARGETS
        self.df = df[keep].dropna().reset_index(drop=True)
        self.root = data_root

        # transforms (center-crop to 256)
        self.tx = transforms.Compose([
            transforms.CenterCrop( min(256,  min(  # be robust to small imgs
                Image.open((self.root/self.df.loc[0,"rgb"]).as_posix()).size)) )  # dummy read to set min side
        ])

        # Proper, deterministic transforms:
        self.tx = transforms.Compose([
            transforms.Resize(288),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

        self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self.root / row["rgb"]
        with Image.open(img_path).convert("RGB") as im:
            x = self.tx(im)
        y = torch.tensor([row[t] for t in TARGETS], dtype=torch.float32)
        return x, y

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_model():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, len(TARGETS))
    return m

def mae(pred, targ):  # for logging
    return (pred - targ).abs().mean().item()

def main():
    device = get_device()
    print(f"Device: {device}")

    train_ds = OverheadDataset(TRAIN_CSV, DATA_ROOT, train=True)
    val_ds   = OverheadDataset(VAL_CSV,   DATA_ROOT, train=False)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2, pin_memory=False)

    model = make_model().to(device)
    crit  = nn.MSELoss()
    opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = math.inf
    epochs = 10
    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        tr_loss, tr_mae = 0.0, 0.0
        for x,y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = crit(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_mae  += mae(pred.detach(), y.detach()) * x.size(0)
        tr_loss /= len(train_ds); tr_mae /= len(train_ds)

        # ---- validate ----
        model.eval()
        va_loss, va_mae = 0.0, 0.0
        with torch.no_grad():
            for x,y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = crit(pred, y)
                va_loss += loss.item() * x.size(0)
                va_mae  += mae(pred, y) * x.size(0)
        va_loss /= len(val_ds); va_mae /= len(val_ds)

        print(f"[ep {ep:02d}] train_mse={tr_loss:.3f}  train_mae={tr_mae:.3f}  "
              f"val_mse={va_loss:.3f}  val_mae={va_mae:.3f}")

        # simple LR decay on plateau
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            torch.save(model.state_dict(), (REPO/"indexes"/"resnet18_overhead_best.pt"))
        else:
            for g in opt.param_groups:
                g["lr"] = max(g["lr"] * 0.5, 1e-5)

    print("Done. Best val MSE:", best_val)

if __name__ == "__main__":
    main()