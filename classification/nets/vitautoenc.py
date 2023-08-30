import math
import torch
from torch import nn
from monai.networks.nets import ViTAutoEnc
from monai.networks.layers import Conv


class EnhancedViTAutoEnc(ViTAutoEnc):
    def __init__(self, feature_dim, return_hidden_states=False, *args, **kwargs):
        super(EnhancedViTAutoEnc, self).__init__(*args, **kwargs)
        self.img_size = kwargs.get('img_size')
        self.out_channels = kwargs.get('out_channels')
        self.feature_dim = feature_dim
        self.return_hidden_states = return_hidden_states
        self.deconv_chns = kwargs.get('deconv_chns', 16)
        self.hidden_size = kwargs.get('hidden_size', 768)

        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        up_kernel_size = [int(math.sqrt(i)) for i in self.patch_size]
        self.conv3d_transpose = conv_trans(
            2 * self.hidden_size, self.deconv_chns, kernel_size=up_kernel_size, stride=up_kernel_size
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=self.deconv_chns, out_channels=self.out_channels, kernel_size=up_kernel_size, stride=up_kernel_size
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_size),
            # Process the feature vector to have the same size as hidden_size
            nn.ReLU(),
        )
        self.final_layer = nn.Linear(self.img_size[0] * self.img_size[1] * self.img_size[2], 1)

    def forward(self, x, features):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        x, features = x.to(torch.float32), features.to(torch.float32)
        spatial_size = x.shape[2:]
        x = self.patch_embedding(x)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])

        fused = self.fusion_layer(features).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([x, fused.expand_as(x)], dim=1)

        x = self.conv3d_transpose(x)
        x = self.conv3d_transpose_1(x)

        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.final_layer(x)
        return (x, hidden_states_out) if self.return_hidden_states else x
