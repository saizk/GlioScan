import torch
import torch.nn.functional as F
from torch import nn
from monai.networks.nets import UNETR


class EnhancedUNETR(UNETR):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_channels = kwargs.get('out_channels')
        self.final_layer = nn.Linear(feature_dim + 1, self.out_channels)

    def forward(self, x_in, features):
        x_in, features = x_in.to(torch.float32), features.to(torch.float32)
        # Previous computations up to vit's output
        # print(x_in.shape, features.shape)
        x, hidden_states_out = self.vit(x_in)
        # print(x.shape)

        # Proceed with the rest of the network computations
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        out = self.out(out)
        # print(out.shape)
        # Late fusion
        features = features.view(features.size(0), features.size(1), 1, 1, 1)
        features = features.expand(-1, -1, *out.shape[2:])
        out = torch.cat([out, features], dim=1)
        # print(out.shape)

        # Global Average Pooling
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))

        # Reduce channel dimension
        out = torch.flatten(out, 1, -1)  # Flatten all dimensions except the batch dimension
        out = self.final_layer(out)  # Use a fully connected layer to reduce channels to 1
        # print(out.shape)
        return out
