import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import VNet


class EnhancedVNet(VNet):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = kwargs.get('out_channels')
        self.final_layer = nn.Linear(feature_dim + 1, self.out_channels)

    def forward(self, x, features):
        x, features = x.to(torch.float32), features.to(torch.float32)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        x = self.up_tr256(out256, out128)
        x = self.up_tr128(x, out64)
        x = self.up_tr64(x, out32)
        x = self.up_tr32(x, out16)
        x = self.out_tr(x)
        # Late fusion
        features = features.view(features.size(0), features.size(1), 1, 1, 1)
        features = features.expand(-1, -1, *x.shape[2:])
        out = torch.cat([x, features], dim=1)

        # Global Average Pooling
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        # Reduce channel dimension
        out = torch.flatten(out, 1, -1)  # Flatten all dimensions except the batch dimension
        out = self.final_layer(out)  # Use a fully connected layer to reduce channels to 1
        return out
