import torch
import torch.nn as nn
from imagenet import VGG16
from embedding import Embed
from text_cnn import TextCnn


class NNModel(nn.Module):

    def __init__(self):
        super(NNModel, self).__init__()
        self.image_net = VGG16()
        self.user_id_embedding = Embed(1009909, 128)
        self.user_type_embedding = Embed(3, 2)
        self.category_embedding = Embed(47, 4)
        self.region_embedding = Embed(28, 4)
        self.city_embedding = Embed(1752, 32)
        self.image_top_embedding = Embed(3064, 32)

        self.text_cnn = TextCnn(300)

        self.classifier = nn.Sequential(
            linear1=nn.Linear(1226, 128),
            bn1=nn.BatchNorm1d(128),
            relu1=nn.ReLU(),
            linear2=nn.Linear(128, 2048),
            bn2=nn.BatchNorm1d(2048),
            relu2=nn.ReLU(),
            linear3=nn.Linear(2048, 128),
            bn3=nn.BatchNorm1d(128),
            relu3=nn.ReLU(),
            linear4=nn.Linear(128, 1),
            sigmoid=nn.Sigmoid()
        )

    def forward(self, feature):
        user_id = self.embedding(feature['user_id'])
        user_type = self.embedding(feature['user_type'])
        category = self.embedding(feature['category'])
        region = self.embedding(feature['region'])
        city = self.embedding(feature['city'])
        image_top = self.embedding(feature['image_top'])

        description = self.text_cnn(feature["description"])
        title = self.text_cnn(feature["description"])

        catted = torch.cat([user_id, user_type, category, region,
                            city, image_top, description, title], dim=1)

        output = self.classifier(catted)
        return output
