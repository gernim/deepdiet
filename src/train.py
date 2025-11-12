import torch
from tqdm import tqdm

from transforms import get_transforms
from src.model import DeepDietModel
from torch.utils.data import DataLoader
from dataset import Nutrition5KDataset


def mae(pred, tgt):
    return torch.mean(torch.abs(pred.squeeze() - tgt).abs())

def main():
    ds = DeepDietModel("data/train_labels.csv", "data/train", transform=get_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    model = DeepDietModel().cuda()
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.9, alpha=0.9, eps=1.0)

    for epoch in range(100):
        model.train()
        for x, y in tqdm(loader):
            x = x.cuda()
            y = {k: v.cuda() for k, v in {k: torch.tensor([y[k] for y in y], dtype=torch.float32) for k in y[0].keys()}.items()}
            out = model(x)
            loss = sum(mae(out[k], y[k]) for k in y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch: {epoch}, loss: {loss}")

if __name__ == "__main__":
    main()