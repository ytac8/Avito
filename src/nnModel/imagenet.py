import torch.nn as nn
from torchvision.models import vgg16_bn, resnet152


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
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


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.model = resnet152(pretrained=True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
