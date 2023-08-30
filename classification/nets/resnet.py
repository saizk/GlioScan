import torch
from torch import nn
from monai.networks.nets import ResNet


class EnhancedResNet(ResNet):
    def __init__(self, feature_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = kwargs.get('num_classes')
        self.final_layer = nn.Linear(feature_dim + 1, self.num_classes)

    def forward(self, x_in, features):
        x_in, features = x_in.to(torch.float32), features.to(torch.float32)
        x = super().forward(x_in)
        out = torch.cat([x, features], dim=1)
        out = self.final_layer(out)  # Flatten all dimensions except the batch dimension
        return out
