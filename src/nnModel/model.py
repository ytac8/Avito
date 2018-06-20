import torch.nn as nn
from imagenet import VGG16
from embedding import Embed


class NNModel(nn.Module):

    def __init__(self):
        super(NNModel, self).__init__()
        self.image_net = VGG16()
        self.embedding = Embed()

        self.classifier = nn.Sequential(
            linear1=nn.Linear(),
            bn1=nn.BatchNorm1d(),
            relu1=nn.ReLU(),
            linear2=nn.Linear(),
            bn2=nn.BatchNorm1d(),
            relu2=nn.ReLU(),
            linear3=nn.Linear(),
            relu3=nn.ReLU(),
            linear4=nn.Linear(),
            sigmoid=nn.Sigmoid()
        )

    def forward(self, feature):
        return output
