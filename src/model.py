import torch
from torchvision import models
from torch import nn


class DeepDietModel(nn.Module):
    """
    EfficientNet-BO inputs:
        Side frames: [B x T, 3, 256, 256]
        We are selecting T (max 16) images from each dish from generated 4 cameras * ~30 images.

        Overhead RGB: [B, 3, 256, 256]

        Overhead Depth: [B, 1, 256, 256]
        Since pretrained weights are based on 3 channels, I convert this to 1 channel by averaging the RGB channels.

    EfficientNet-BO outputs:
        [B, T, 1280]
        Classifier class logits are removed and feature vector is used as output.

    BiLSTM inputs:
        Side frames: [B x T, 1280]

    """

    def __init__(self, use_side_frames=True, use_overhead=True, use_depth=True, chunk_size=4, lstm_hidden=640):
        super().__init__()
        self.use_side_frames = use_side_frames
        self.use_overhead = use_overhead
        self.use_depth = use_depth and use_overhead
        self.chunk_size = chunk_size
        self.lstm_hidden = lstm_hidden

        total_features = 0

        if self.use_side_frames:
            self.side_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.side_encoder.classifier = nn.Identity()
            self.side_aggregator = nn.LSTM(1280, self.lstm_hidden, batch_first=True, bidirectional=True)
            total_features += self.lstm_hidden * 2

        if self.use_overhead:
            self.overhead_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.overhead_encoder.classifier = nn.Identity()
            total_features += 1280

        if self.use_depth:
            self.depth_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            # Since pretrained weights are based on 3 channels, convert this to 1 channel by averaging the RGB channels.
            original_conv = self.depth_encoder.features[0][0]
            self.depth_encoder.features[0][0] = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, stride=original_conv.stride, padding=original_conv.padding, bias=False)
            self.depth_encoder.classifier = nn.Identity()
            total_features += 1280

        self.fusion_layers = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.mass_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.calorie_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        self.macro_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, side_frames, overhead_rgb=None, overhead_depth=None):
        features = []

        if self.use_side_frames and side_frames is not None:
            batch_size = side_frames.size(0)
            num_frames = side_frames.size(1)

            # Flatten: [B, T, C, H, W] to [B * T, C, H, W]
            side_frames_flat = side_frames.view(batch_size * num_frames, *side_frames.shape[2:])
            side_features = []

            for i in range(0, side_frames_flat.size(0), self.chunk_size):
                chunk = side_frames_flat[i:i + self.chunk_size]
                chunk_feat = self.side_encoder(chunk)
                side_features.append(chunk_feat)

            # Reshape for LSTM: [B * T, 1280] -> [B, T, 1280]
            side_features = torch.cat(side_features, dim=0).view(batch_size, num_frames, -1)


            side_feat, _ = self.side_aggregator(side_features)
            side_feat = side_feat.mean(dim=1)
            features.append(side_feat)

        if self.use_overhead and overhead_rgb is not None:
            features.append(self.overhead_encoder(overhead_rgb))

        if self.use_depth and overhead_depth is not None:
            features.append(self.depth_encoder(overhead_depth))

        concat_features = torch.cat(features, dim=1)
        fused_features = self.fusion_layers(concat_features)
        mass = self.mass_head(fused_features)
        calories = self.calorie_head(fused_features)
        macros = self.macro_head(fused_features)

        return torch.cat([calories, mass, macros], dim=1)