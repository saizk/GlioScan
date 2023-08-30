import torch
from torch import nn
from collections.abc import Sequence
from collections import OrderedDict
from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.nets.densenet import _DenseBlock, _Transition


class EnhancedDenseNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
        dropout_prob: float = 0.0,
        feature_dim: int = 0
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
        Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )
        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.features.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.features.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.features.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        # pooling and classification
        # flattened_size = in_channels + feature_dim
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    # ("out", nn.Linear(in_channels, out_channels)),
                ]
            )
        )
        self.final_layer = nn.Linear(in_channels + feature_dim, out_channels)

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor, feature_vector: torch.Tensor) -> torch.Tensor:
        x, feature_vector = x.to(torch.float32), feature_vector.to(torch.float32)
        # print(x.shape, feature_vector.shape)
        x = self.features(x)
        # print(x.shape)
        x = self.class_layers(x)
        # print(x.shape)
        # x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = torch.cat([x, feature_vector], dim=1)  # Concatenate along dimension 1 (channels)
        # x = self.class_layers(x)
        x = x.to(torch.float32)
        x = self.final_layer(x)
        return x
