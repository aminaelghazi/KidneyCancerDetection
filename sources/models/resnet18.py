import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ModifiedResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18(weights=None)
        # Modify first conv to accept single channel (CT grayscale)
        original_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, original_conv.out_channels,
                                     kernel_size=original_conv.kernel_size,
                                     stride=original_conv.stride,
                                     padding=original_conv.padding,
                                     bias=False)
        # Initialize new conv weights from original (average across RGB channels)
        with torch.no_grad():
            self.model.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        # Replace final FC layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
