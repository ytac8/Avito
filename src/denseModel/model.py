import torch.nn as nn
import torch.nn.functional as f


class CountFeatureNN(nn.Module):

    def __init__(self, num_feature, out_size):
        super(CountFeatureNN, self).__init__()
        self.fc1 = nn.Linear(num_feature, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, out_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.fc1(input)
        output = self.bn1(output)
        output = f.relu(output)
        output = self.fc2(output)
        output = self.bn2(output)
        output = f.relu(output)
        output = self.fc3(output)
        output = self.bn3(output)
        output = f.relu(output)
        output = self.fc4(output)
        output = self.softmax(output)
        return output
