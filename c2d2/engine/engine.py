import chess
from c2d2.engine.evaluator import Evaluator

def count_pieces(board):
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

    material = 100*(wp-bp)+320*(wn-bn)+330*(wb-bb)+500*(wr-br)+900*(wq-bq)

    return material


class Engine():
    def __init__(self, depth):
        self.depth = depth
        self.board = chess.Board()
        self.evaluator = Evaluator()

    def make_move(self, uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)


    def nega_max(self, board, depth, color):
        
        if depth == 0 or board.is_game_over():
            return [self.evaluator.eval(board), None]

        best = [-9999, None]
        for move in board.legal_moves:
            board.push(move)
            score = -1 * self.nega_max(board, depth - 1, -1 * color)[0]
            if score > best[0]:
                best = [score, move]

            board.pop()

        return best


    def find_best_move(self):
        material = count_pieces(self.board)

        turn = 1 if self.board.turn else -1

        score, move = self.nega_max(self.board, self.depth, turn)
        return move, score
    
