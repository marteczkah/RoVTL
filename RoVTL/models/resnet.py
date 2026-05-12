import torch
import torch.nn as nn
from monai.networks.nets import ResNetFeatures
import torchvision.models as models

class ResNET(nn.Module):
    def __init__(self, config, dim, in_channels=11):
        super().__init__()
        self.dim = dim
        if dim == 3:
            self.resnet = ResNetFeatures('resnet50', pretrained=False, spatial_dims=3, in_channels=in_channels)
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  
        else:
            resnet = models.resnet50(weights=None, num_classes=1)
            self.resnet = nn.Sequential(*list(resnet.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.dim_projection = config['dim_projection']
        self.projection_heads = torch.nn.ModuleDict(
            {
                f"proj_{d_proj}": nn.Sequential(
                    nn.Linear(2048, d_proj),
                    nn.Dropout(config['dropout']),
                    nn.ReLU(),
                    nn.LayerNorm(d_proj),
                    nn.Linear(d_proj, config['dim_feedforward']),
                    nn.ReLU(),
                    nn.LayerNorm(config['dim_feedforward']),
                    nn.Linear(config['dim_feedforward'], d_proj),
                )
                for d_proj in self.dim_projection
            }
        )    

    def forward(self, x):
        if self.dim == 3:
            out = self.resnet(x)[-1]  # (batch_size, 2048, D, H, W)
        else:
            out = self.resnet(x)
        feature = self.gap(out).squeeze()  # (batch_size, 2048)
        if len(feature.shape) == 1:
            feature = feature.unsqueeze(0)
        projections = []
        for lin_proj in self.projection_heads.values():
            proj = lin_proj(feature)
            projections.append(proj)
            
        return projections, feature