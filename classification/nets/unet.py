import torch
import torch.nn.functional as F
from torch import nn
from monai.networks.nets import Unet


class EnhancedUNET(Unet):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = kwargs.get('out_channels')
        self.final_layer = nn.Linear(feature_dim + 1, self.out_channels)

    def forward(self, x_in, features):
        x_in, features = x_in.to(torch.float32), features.to(torch.float32)
        x = self.model(x_in)
        features = features.view(features.size(0), features.size(1), 1, 1, 1)
        features = features.expand(-1, -1, *x.shape[2:])
        out = torch.cat([x, features], dim=1)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))  # Global Average Pooling
        # Reduce channel dimension
        out = torch.flatten(out, 1, -1)  # Flatten all dimensions except the batch dimension
        out = self.final_layer(out)
        return out
