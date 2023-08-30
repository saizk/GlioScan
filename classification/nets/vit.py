import torch
from torch import nn
from monai.networks.nets import ViT


class EnhancedViT(ViT):
    def __init__(self, feature_dim, return_hidden_states=False, *args, **kwargs):
        super(EnhancedViT, self).__init__(*args, **kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = kwargs.get('hidden_size', 384)
        # Linear layer to transform the feature vector to a suitable dimension
        self.feature_transform = nn.Linear(self.feature_dim, self.hidden_size)
        self.return_hidden_states = return_hidden_states

    def forward(self, x, features):
        x, features = x.to(torch.float32), features.to(torch.float32)

        # First, compute the patch embeddings from x
        x = self.patch_embedding(x)

        # Transform the feature vector
        features = self.feature_transform(features).unsqueeze(1)  # Increase the sequence length by 1 <-- NEW
        # Concatenate the feature vector with the patch embeddings along the sequence dimension<-- NEW
        x = torch.cat((features, x), dim=1)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)

        x = self.norm(x)
        if hasattr(self, "classification_head"):
            x = self.classification_head(x[:, 0])

        return (x, hidden_states_out) if self.return_hidden_states else x
