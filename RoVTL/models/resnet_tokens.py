import torch.nn as nn
from monai.networks.nets import ResNetFeatures
import torchvision.models as models

class ResNetTokens(nn.Module):
    def __init__(self, dim, in_channels=11):
        super().__init__()
        self.dim = dim
        if self.dim == 3:
            self.resnet = ResNetFeatures('resnet50', pretrained=False, spatial_dims=3, in_channels=in_channels)
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))  
        else:
            resnet = models.resnet50(weights=None, num_classes=1)
            self.resnet = nn.Sequential(*list(resnet.children())[:-2])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        if self.dim == 3:
            out = self.resnet(x)[-1]
        else:
            out = self.resnet(x)
        if out.shape[0] == 1:
            tokens = out.squeeze().unsqueeze(0).flatten(start_dim=2) 
            tokens = tokens.permute(0, 2, 1)
        else:
            tokens = out.squeeze().flatten(start_dim=2) 
            tokens = tokens.permute(0, 2, 1)
        feature = self.gap(out).squeeze() 
        if len(feature.shape) == 1:
            feature = feature.unsqueeze(0)
        return feature, tokens