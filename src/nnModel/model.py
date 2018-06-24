import torch
import torch.nn as nn
from imagenet import VGG16
from text_cnn import TextCnn


class NNModel(nn.Module):

    def __init__(self):
        super(NNModel, self).__init__()
        self.image_net = VGG16()
        self.user_id_embedding = nn.Embedding(1009909, 128)
        self.user_type_embedding = nn.Embedding(3, 2)
        self.category_embedding = nn.Embedding(47, 4)
        self.region_embedding = nn.Embedding(28, 4)
        self.city_embedding = nn.Embedding(1752, 32)
        self.image_top_embedding = nn.Embedding(3064, 32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.text_cnn = TextCnn(300)

        self.classifier = nn.Sequential(
            nn.Linear(1226, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        user_id = self.user_id_embedding(
            feature['user_id'].to(self.device)).squeeze()
        user_type = self.user_type_embedding(
            feature['user_type'].to(self.device)).squeeze()
        category = self.category_embedding(
            feature['category'].to(self.device)).squeeze()
        region = self.region_embedding(
            feature['region'].to(self.device)).squeeze()
        city = self.city_embedding(feature['city'].to(self.device)).squeeze()
        image_top = self.image_top_embedding(
            feature['image_top'].to(self.device)).squeeze()

        description = self.text_cnn(feature["description"].to(self.device))
        title = self.text_cnn(feature["title"].to(self.device))

        catted = torch.cat((user_id, user_type, category, region,
                            city, image_top, description, title), 1)

        output = self.classifier(catted)
        return output
