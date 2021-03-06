import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.bfc1 = nn.Linear(768, 2048)
        self.bfc2 = nn.Linear(2048, 2048)
        self.bfc3 = nn.Linear(2048, 2048)

        self.bDrop1 = nn.Dropout(p=0.5)
        self.bDrop2 = nn.Dropout(p=0.5)

        self.ffc1 = nn.Linear(3, 32)

        self.fc1 = nn.Linear(2080, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(2048, 1)

    def forward(self, board, features):
        board = self.bDrop1(F.relu(self.bfc1(board)))
        board = self.bDrop2(F.relu(self.bfc2(board)))
        board = F.relu(self.bfc3(board))

        features = F.relu(self.ffc1(features))
        x = torch.cat((board, features), 1)

        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))

        x = self.drop3(self.bn3(F.relu(self.fc3(x))))
        x = self.fc4(x)
        return x

