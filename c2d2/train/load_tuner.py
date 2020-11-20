import chess
import chess.engine
import numpy as np
import torch

from c2d2.train.modelInput import genX

class TunerPositions():
    def __init__(self, path, batch_size, depth):
        self.path = path
        self.depth = depth
        self.batch_size = batch_size
        self.stockfish = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

        #result = engine.analyse(board, chess.engine.Limit(depth=depth))
        #engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    
    def stockfishEval(self, fen):
        board = chess.Board(fen)
        res = self.stockfish.analyse(board, chess.engine.Limit(depth=self.depth))
        y = res['score'].white() if board.turn else res['score'].black()

        try:
            y = float(str(y))
        except ValueError:
            y = None

        return y



    def batches(self):
        dataset = open(self.path, 'r')

        board_buffer = []
        feature_buffer = []
        y_buffer = []

        for fen in dataset:
            y = self.stockfishEval(fen)
            if y is None: 
                continue
            y /= 100

            brd, features = genX(fen)

            board_buffer.append(brd)
            feature_buffer.append(features)
            y_buffer.append(y)

            if len(board_buffer) == self.batch_size:
                boards = torch.Tensor(board_buffer)
                features = torch.Tensor(feature_buffer)
                ys = torch.Tensor(y_buffer)

                board_buffer = []
                feature_buffer = []
                y_buffer = []
           
                yield boards, features, ys

        dataset.close()

    
    def close(self):
        self.stockfish.close()

if __name__ == '__main__':
    data = TunerPositions('../quiet.epd', 512, 0)

    for brd, features, y in data.batches():
        print(brd.shape, features.shape, y.shape)

    data.close()
