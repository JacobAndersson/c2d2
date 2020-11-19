import torch
import torch.nn as nn
import torch.nn.functional as F

class fusionModel(nn.Module):
    def __init__(self):
        super(fusionModel, self).__init__()

        self.v1 = nn.Linear(64, 1024)
        self.v2 = nn.Linear(1024, 1024)
        self.vb1 = nn.BatchNorm1d(1024)
        self.vb2 = nn.BatchNorm1d(1024)

        self.f1 = nn.Linear(2, 32)
        self.f2 = nn.Linear(32, 32)
        self.fb1 = nn.BatchNorm1d(32)
        self.fb2 = nn.BatchNorm1d(32) 

        self.fc1 = nn.Linear(1056, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, vector, features):
        vector = F.relu(self.vb1(self.v1(vector)))
        vector = F.relu(self.vb2(self.v2(vector)))

        features = F.relu(self.fb1(self.f1(features)))
        features = F.relu(self.fb2(self.f2(features)))

        x = torch.cat((vector, features), 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


