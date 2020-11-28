import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

from c2d2.train.models.simpleModel import SimpleModel 
from c2d2.train.models.smaller import Smaller
from c2d2.train.models.quant import Quant

from c2d2.train.load_data import chessPositions
from c2d2.train.load_tuner import TunerPositions

device = torch.device(0 if torch.cuda.is_available() else 'cpu')
EPOCH_PATH = './c2d2/train/checkpoints/model_smaller_epoch_{}.pt'
BATCH_PATH = './c2d2/train/checkpoints/model_smaller_batch_{}.pt'

model = Quant()
model.to(device)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(dataset):
    model.train()
    running_loss = 0.0
    running_mae = 0.0 

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

        mae = F.l1_loss(y_pred, y)
        running_mae += mae.item()

        print('{}, LOSS: {}'.format(idx, loss.item()))
        if (idx % 200 == 0):
            print('{}, RUNNING LOSS: {}'.format(idx, running_loss/(idx+1)))

        if (idx%50000 == 0):
            pth = BATCH_PATH.format(idx)
            torch.save(model.state_dict(), pth)

    
    avg_loss = running_loss/(idx + 1)
    avg_mae = running_mae/(idx+1)

    print('FINAL AVERAGE: {}, MAE: {}'.format(avg_loss, avg_mae))
    return avg_mae, avg_loss

def test_epoch(dataset):
    model.eval()
    running_loss = 0.0
    running_mae = 0.0
    for idx, (board, features, y) in enumerate(dataset.batches()):
        board = board.to(device)
        features = features.to(device)
        y = y.unsqueeze(1).to(device)

        y_pred = model(board, features)
        loss = criterion(y_pred, y)
        running_loss += loss.item()

        mae = F.l1_loss(y_pred, y)
        running_mae += mae.item()

        print(idx, loss.item())

    avg_mse = running_loss / (idx+1)
    avg_mae = running_mae / (idx+1)
    print("EPOCH AVERAGE: {}, MAE: {}".format(avg_mse, avg_mae))

    return avg_mae, avg_mse

def start(num_epochs, min_num_epochs, batch_size):
    #dataset = chessPositions('./c2d2/dataset_zero.csv', 2048, 0)
    #test = chessPositions('./c2d2/dataset_zero_test.csv', 2048, 0)
    dataset = TunerPositions('./c2d2/quiet.epd', batch_size, 0) 
    test = TunerPositions('./c2d2/quiet_test.epd', batch_size, 0) 
    
    train_loss = []
    test_loss = []
    for epoch in range(num_epochs):
        print("EPOCH: {}".format(epoch))
        print('####TRAIN####')
        train_mae, train_mse = train_epoch(dataset)
        train_loss.append((train_mae, train_mse))

        pth = EPOCH_PATH.format(epoch)
        torch.save(model.state_dict(), pth)
        print('saved model: {}'.format(pth))
        print('#####TEST####')
        test_mae, test_mse = test_epoch(test)
        test_loss.append((test_mae, test_mse))

        if epoch > min_num_epochs:
            if test_mse > test_loss[-3][1]:
                print('stopped early', epoch)
                break
    
    print('\n\nTRAIN')
    print(train_loss)
    print('TEST')
    print(test_loss)
    dataset.close()
    test.close()
