import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCnn(nn.Module):

    def __init__(self, emb_dim):

        super(TextCnn, self).__init__()
        self.cnn1 = nn.Conv2d(1, 32, (1, emb_dim))
        self.cnn2 = nn.Conv2d(1, 32, (2, emb_dim))
        self.cnn3 = nn.Conv2d(1, 32, (3, emb_dim))
        self.cnn4 = nn.Conv2d(1, 32, (4, emb_dim))

        self.classifier = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

    def forward(self, input):
        batch_size = input.size(0)

        # input = (B * C * H * W)
        output1 = self.cnn1(input)
        output2 = self.cnn2(input)
        output3 = self.cnn3(input)
        output4 = self.cnn4(input)

        # input = (B * 32 * H * W)
        output1 = F.relu(F.max_pool2d(output1, (output1.size(2), 1)))
        output2 = F.relu(F.max_pool2d(output2, (output2.size(2), 1)))
        output3 = F.relu(F.max_pool2d(output3, (output3.size(2), 1)))
        output4 = F.relu(F.max_pool2d(output4, (output4.size(2), 1)))

        output1 = output1.view(batch_size, -1)
        output2 = output2.view(batch_size, -1)
        output3 = output3.view(batch_size, -1)
        output4 = output4.view(batch_size, -1)

        catted = torch.cat((output1, output2, output3, output4), 1)
        output = self.classifier(catted)
        return output
