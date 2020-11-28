import torch 
import torch.nn as nn
import torch.nn.functional as F

class Smaller(nn.Module):
    def __init__(self):
        super(Smaller, self).__init__()
        self.bfc1 = nn.Linear(768, 1024)
        self.bfc2 = nn.Linear(1024, 1024)

        self.bDrop1 = nn.Dropout(p=0.5)

        self.ffc1 = nn.Linear(3, 32)

        self.fc1 = nn.Linear(1056, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(1024, 1)

    def forward(self, board, features):
        board = self.bDrop1(F.relu(self.bfc1(board)))
        board = F.relu(self.bfc2(board))

        features = F.relu(self.ffc1(features))
        x = torch.cat((board, features), 1)
        
        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc4(x)
        return x

