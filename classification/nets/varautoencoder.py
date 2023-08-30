import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import VarAutoEncoder


class EnhancedVarAutoEncoder(VarAutoEncoder):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sigmoid = False
        self.final_layer = nn.Linear(feature_dim + 1, self.out_channels)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x, features = x.to(torch.float32), features.to(torch.float32)
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode_forward(z, self.use_sigmoid)
        # assert not torch.isnan(out).any()

        features = features.view(features.size(0), features.size(1), 1, 1, 1)
        features = features.expand(-1, -1, *x.shape[2:])

        out = torch.cat([out, features], dim=1)

        # Global Average Pooling
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        # Reduce channel dimension
        out = torch.flatten(out, 1, -1)  # Flatten all dimensions except the batch dimension
        out = self.final_layer(out)  # Use a fully connected layer to reduce channels to 1
        # assert not torch.isnan(out).any()
        return out
