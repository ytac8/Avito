import torch.nn as nn
import torch


class CharCNN(nn.Module):

    def __init__(self, feature_dim, dropout_p, output_size=2, filter_num=256):
        super(CharCNN, self).__init__()

        self.output_size = output_size
        self.filter_num = filter_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, filter_num, kernel_size=(7, feature_dim), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(1, filter_num, kernel_size=(7, filter_num), stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(1, filter_num, kernel_size=(3, filter_num), stride=1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(29184, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

        self.fc3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.conv1(input)
        output = torch.transpose(output, 1, 3)
        output = self.conv2(output)
        output = torch.transpose(output, 1, 3)
        output = self.conv3(output)
        output = torch.transpose(output, 1, 3)
        output = self.conv3(output)
        output = torch.transpose(output, 1, 3)
        output = self.conv3(output)
        output = torch.transpose(output, 1, 3)
        output = self.conv2(output)
        output = torch.transpose(output, 1, 3)

        output = output.contiguous().view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.sigmoid(output)

        return output
