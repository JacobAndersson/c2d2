import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

from c2d2.train.models.simpleModel import SimpleModel 
from c2d2.train.load_data import chessPositions

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
EPOCH_PATH = './c2d2/train/checkpoints/model_epoch_{}.pt'
BATCH_PATH = './c2d2/train/checkpoints/model_batch_{}.pt'

model = SimpleModel()
model.to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(dataset):
    model.train()
    running_loss = 0.0
    for idx, (board, features, y) in enumerate(dataset.batches()):
        optimizer.zero_grad()
        board = board.to(device)
        features = features.to(device)
        y = y.unsqueeze(1).to(device)

        y_pred = model(board, features)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (idx%50 == 0):
            print('{}, RUNNING LOSS: {}'.format(idx, running_loss/(50)))
            running_loss = 0.0

        if (idx%50000  == 0):
            pth = BATCH_PATH.format(idx)
            torch.save(model.state_dict(), pth)

def test_epoch(dataset):
    model.eval()
    running_loss = 0.0
    for idx, (board, features, y) in enumerate(dataset.batches()):
        board = board.to(device)
        features = features.to(device)
        y = y.unsqueeze(1).to(device)

        y_pred = model(board, features)
        loss = criterion(y_pred, y)
        
        running_loss += loss.item()
        if (idx%10 == 0):
            print(idx, running_loss/(idx + 1))
            running_loss = 0.0

def start():
    dataset = chessPositions('./c2d2/dataset.csv', 2048, 0)
    test = chessPositions('./c2d2/test_dataset.csv', 2048, 0)

    for epoch in range(8):
        print('####TRAIN####')
        train_epoch(dataset)
        pth = EPOCH_PATH.format(epoch)
        torch.save(model.state_dict(), pth)
        print('saved model: {}'.format(pth))
        print('#####TEST####')
        test_epoch(test)
    
