import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

class Quant(nn.Module):
    def __init__(self):
        super(Quant, self).__init__()
        self.bfc1 = nn.Linear(768, 512)
        self.bfc2 = nn.Linear(512, 512)

        self.bDrop1 = nn.Dropout(p=0.5)

        self.ffc1 = nn.Linear(3, 32)

        self.fc1 = nn.Linear(544, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

        self.fc4 = nn.Linear(512, 1)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, board, features):
        board = self.quant(board)
        board = self.bDrop1(F.relu(self.bfc1(board)))
        board = F.relu(self.bfc2(board))

        features = self.quant(features)
        features = F.relu(self.ffc1(features))
        x = torch.cat((board, features), 1)

        x = self.drop1(self.bn1(F.relu(self.fc1(x))))
        x = self.drop2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc4(x)
        x = self.dequant(x)
        return x

