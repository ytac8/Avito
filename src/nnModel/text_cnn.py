import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnn(nn.Module):

    def __init__(self, emb_dim):

        super(TextCnn, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, (emb_dim, 1))
        self.cnn2 = nn.Conv2d(1, 32, (emb_dim, 2))
        self.cnn3 = nn.Conv2d(1, 32, (emb_dim, 3))
        self.cnn4 = nn.Conv2d(1, 32, (emb_dim, 4))

        self.classifier = nn.Sequential(
            linear1=nn.Linear(128, 1024),
            bn1=nn.BatchNorm1d(1024),
            relu1=nn.ReLU(),
            linear2=nn.Linear(1024, 512),
            bn2=nn.BatchNorm1d(512),
            relu2=nn.ReLU()
        )

    def forward(self, input):
        output1 = self.cnn1(input).squeeze()
        output2 = self.cnn2(input).squeeze()
        output3 = self.cnn3(input).squeeze()
        output4 = self.cnn4(input).squeeze()

        output1 = F.relu(F.max_pool1d(output1).squeeze())
        output2 = F.relu(F.max_pool1d(output2).squeeze())
        output3 = F.relu(F.max_pool1d(output3).squeeze())
        output4 = F.relu(F.max_pool1d(output4).squeeze())

        catted = torch.cat([output1, output2, output3, output4], dim=1)
        output = self.classifier(catted)
        return output
