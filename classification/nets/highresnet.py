from collections.abc import Sequence

import torch
from torch import nn
from monai.utils import ChannelMatching
from monai.networks.nets import HighResNet
from monai.networks.nets.highresnet import DEFAULT_LAYER_PARAMS_3D


class EnhancedHighResNet(HighResNet):

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        feature_dim: int = 128,  # New feature dimension parameter
        norm_type: str | tuple = ("batch", {"affine": True}),
        acti_type: str | tuple = ("relu", {"inplace": True}),
        dropout_prob: tuple | str | float | None = 0.0,
        bias: bool = False,
        layer_params: Sequence[dict] = DEFAULT_LAYER_PARAMS_3D,
        channel_matching: ChannelMatching | str = ChannelMatching.PAD,
    ) -> None:
        super().__init__(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                         norm_type=norm_type, acti_type=acti_type, dropout_prob=dropout_prob,
                         bias=bias, layer_params=layer_params, channel_matching=channel_matching)

        # New layer to process the feature vector
        self.fusion_layer = nn.Linear(in_channels + feature_dim, out_channels)

    def forward(self, x: torch.Tensor, feature_vector: torch.Tensor) -> torch.Tensor:
        x, feature_vector = x.to(torch.float32), feature_vector.to(torch.float32)

        x = self.blocks(x)
        # Average pooling across spatial dimensions to reduce to a vector per batch
        avg_pool = nn.AvgPool3d(kernel_size=x.size()[2:])
        x = avg_pool(x).view(x.size(0), -1)
        # print(x.shape, feature_vector.shape)
        # Concatenate feature vector and output from the network
        x = torch.cat((x, feature_vector), dim=1)
        # print(x.shape)
        # Pass through the fusion layer
        x = self.fusion_layer(x)
        return torch.as_tensor(x)
