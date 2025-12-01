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

    def __init__(self, use_side_frames=True, use_overhead=True, use_depth=True, chunk_size=4, lstm_hidden=640, side_aggregation='lstm'):
        super().__init__()
        self.use_side_frames = use_side_frames
        self.use_overhead = use_overhead
        self.use_depth = use_depth and use_overhead
        self.chunk_size = chunk_size
        self.lstm_hidden = lstm_hidden
        self.side_aggregation = side_aggregation  # 'lstm' or 'mean'

        total_features = 0

        if self.use_side_frames:
            self.side_encoder = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.side_encoder.classifier = nn.Identity()

            if self.side_aggregation == 'lstm':
                self.side_aggregator = nn.LSTM(1280, self.lstm_hidden, batch_first=True, bidirectional=True)
                total_features += self.lstm_hidden * 2
            else:
                # Mean pooling: output is raw EfficientNet features (1280)
                total_features += 1280

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

    def _get_encoders(self):
        """Return list of encoder modules that exist."""
        encoders = []
        if self.use_side_frames and hasattr(self, 'side_encoder'):
            encoders.append(self.side_encoder)
        if self.use_overhead and hasattr(self, 'overhead_encoder'):
            encoders.append(self.overhead_encoder)
        if self.use_depth and hasattr(self, 'depth_encoder'):
            encoders.append(self.depth_encoder)
        return encoders

    def freeze_encoders(self):
        """Freeze all EfficientNet encoder weights."""
        for encoder in self._get_encoders():
            for param in encoder.parameters():
                param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze all EfficientNet encoder weights."""
        for encoder in self._get_encoders():
            for param in encoder.parameters():
                param.requires_grad = True

    def get_param_groups(self, base_lr, encoder_lr_multiplier=0.1):
        """
        Return parameter groups with different learning rates.
        - Encoder params: base_lr * encoder_lr_multiplier
        - Other params (LSTM, fusion, heads): base_lr
        """
        encoder_params = []
        other_params = []

        encoder_modules = set(self._get_encoders())

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            # Check if this param belongs to an encoder
            is_encoder_param = any(
                name.startswith(enc_name)
                for enc_name in ['side_encoder', 'overhead_encoder', 'depth_encoder']
            )

            if is_encoder_param:
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': other_params, 'lr': base_lr},
        ]

        # Only add encoder group if there are unfrozen encoder params
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': base_lr * encoder_lr_multiplier
            })

        return param_groups

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

            # Reshape: [B * T, 1280] -> [B, T, 1280]
            side_features = torch.cat(side_features, dim=0).view(batch_size, num_frames, -1)

            # Aggregate frame features
            if self.side_aggregation == 'lstm':
                side_feat, _ = self.side_aggregator(side_features)
                side_feat = side_feat.mean(dim=1)
            else:
                # Simple mean pooling over frames
                side_feat = side_features.mean(dim=1)

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