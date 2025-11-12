from pyexpat import features

import torchvision.models as M
from numpy.random import weibull
from torch import nn


class DeepDietModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = M.inception_v3()
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d((8,8)), nn.Flatten(),
            nn.Linear(features, 4086), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU()
        )

        def head(features):
            return nn.Sequential(
                nn.Linear(4096, 4096), nn.ReLU(),
                nn.Linear(4096, 1)
            )
        self.cal = head();
        self.mass = head();
        self.fat = head();
        self.carb = head();
        self.protein = head();

    def forward(self, x):
        f = self.shared(self.backbone(x))

        return dict(
            cal = self.cal(f),
            mass = self.mass(f),
            fat = self.fat(f),
            carb = self.carb(f),
            protein = self.protein(f)
        )