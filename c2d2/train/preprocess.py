from torch.utils.data import Dataset
import chess
import numpy as np

class chessPositions(Dataset):
    def __init__(self, data):
        self.data = data
        self.piece2Idx = {
            'r': 1, 'b': 2, 'k': 3,
            'q': 4, 'n': 5, 'p': 6,
            'P': 7, 'R': 8, 'N': 9,
            'B': 10, 'K': 11, 'Q': 12
        }

    def __getitem__(self,idx):
        fen, y = self.data[idx]
        vector, features = self.genX(fen)
        return vector, features, y

    def __len__(self):
        return self.data.shape[0]

    def genX(self, fen):
        board = chess.Board(fen)
        vector = self.getBoardRep(board)
        handCrafted = self.handCraftedFeatures(board)

        return vector, handCrafted

    def getBoardRep(self, board):
        pieces = board.piece_map()
        vector = [0 for _ in range(64)]

        for (idx, piece) in pieces.items():
            symbol = piece.symbol()
            num = self.piece2Idx[symbol]
            vector[idx] = num

        return np.array(vector)


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
        turn = 1 if board.turn else 1
        return np.array([material, attacked, turn])


