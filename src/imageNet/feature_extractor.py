import torch.nn as nn
from torchvision.models import vgg16_bn


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        original_model = vgg16_bn(pretrained=True)
        self.features = nn.Sequential(
            *list(original_model.features.children())
        )
        self.avg_pool = nn.AvgPool2d(7)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.squeeze()
        return x
