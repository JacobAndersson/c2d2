import chess

class Engine():
    def __init__(self):
        self.board = chess.Board()

    def make_move(self, uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)
    
