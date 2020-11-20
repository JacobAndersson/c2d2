import chess
import numpy as np
import torch

class chessPositions():
    def __init__(self, path, batch_size, index):
        #fen,depth3,depth4,depth6,depth8
        self.path = path
        self.batch_size = batch_size
        self.index = index
        

        self.piece2Idx = {
            'r': 1, 'b': 2, 'k': 3,
            'q': 4, 'n': 5, 'p': 6,
            'P': 7, 'R': 8, 'N': 9,
            'B': 10, 'K': 11, 'Q': 12
        }

    def batches(self):
        dataset = open(self.path, 'r')
        next(dataset) #skip header
        
        board_buffer = []
        feature_buffer = []
        y_buffer = []
        for line in dataset:
            row = line.split(',')
            
            fen = row[0]
            board, features = self.genX(fen)
            y = torch.Tensor([float(row[1:][self.index]) / 100])
            
            board_buffer.append(board)
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

        

    def genX(self, fen):
        board = chess.Board(fen)
        vector = self.getBoardRep(board)
        handCrafted = self.handCraftedFeatures(board)

        return vector, handCrafted

    def getBoardRep(self, board):
        pieces = board.piece_map()
        vector = [0 for _ in range(64)]

        boardVec = np.array([[0 for _ in range(12)] for _ in range(64)])
        boardVec[63, 0] = 1

        for (idx, piece) in pieces.items():
            symbol = piece.symbol()
            num = self.piece2Idx[symbol]
            boardVec[idx, num-1] = 1
        
        boardVec = boardVec.flatten()
        return boardVec


    def getMaterial(self, board):
        wp = len(board.pieces(chess.PAWN, chess.WHITE))
        bp = len(board.pieces(chess.PAWN, chess.BLACK))
        wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(board.pieces(chess.ROOK, chess.WHITE))
        br = len(board.pieces(chess.ROOK, chess.BLACK))
        wq = len(board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(board.pieces(chess.QUEEN, chess.BLACK))

        total_material = bp + bn + bb + br + bq + wp + wn + wb + wr + wq
        return total_material

    def countAttacked(self, board):
        pieces = board.piece_map()
        idxs = list(pieces.keys())

        white_attacked = 0
        black_attacked = 0

        for idx in idxs:
            square = chess.SQUARE_NAMES[idx]

            #True - white, false - black
            attack_color = chess.WHITE if not board.color_at(idx) else chess.BLACK

            num_attackers = len(board.attackers(attack_color, idx))

            if num_attackers > 0:
                if not attack_color:
                    white_attacked += 1
                else:
                    black_attacked += 1

        return black_attacked - white_attacked

    def handCraftedFeatures(self, board):
        material = self.getMaterial(board)
        attacked = self.countAttacked(board)
        turn = 1 if board.turn else 0
        return np.array([material, attacked, turn])
