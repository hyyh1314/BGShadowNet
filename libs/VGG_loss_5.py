
import torch.nn as nn
from torchvision import models
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['3', '8','13','22','31']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():

            x = layer(x)
            if name in self.select:
                features.append(x)
        return features[0], features[1],features[2],features[3],features[4]