import chess
from c2d2.engine.evaluator import Evaluator
from c2d2.engine import search
from c2d2.engine.piece_tables import *

class material():
    def eval(self, board):
        turn = 1 if board.turn else -1
        if (board.is_checkmate()):
            return turn * 9000

        return self.getMaterial(board) + self.piece_Tables(board)

    def getMaterial(self, board):
        wp = board.pieces(chess.PAWN, chess.WHITE)
        bp = board.pieces(chess.PAWN, chess.BLACK)
        wn = board.pieces(chess.KNIGHT, chess.WHITE)
        bn = board.pieces(chess.KNIGHT, chess.BLACK)
        wb = board.pieces(chess.BISHOP, chess.WHITE)
        bb = board.pieces(chess.BISHOP, chess.BLACK)
        wr = board.pieces(chess.ROOK, chess.WHITE)
        br = board.pieces(chess.ROOK, chess.BLACK)
        wq = board.pieces(chess.QUEEN, chess.WHITE)
        bq = board.pieces(chess.QUEEN, chess.BLACK)
        wk = board.pieces(chess.KING, chess.WHITE)
        bk = board.pieces(chess.KING, chess.BLACK)

        material = 100*(len(wp)-len(bp))+320*(len(wn)-len(bn))+330*(len(wb)-len(bb))+500*(len(wr)-len(br))+900*(len(wq)-len(bq))

        return material
    def piece_Tables(self, board):
        wp = board.pieces(chess.PAWN, chess.WHITE)
        bp = board.pieces(chess.PAWN, chess.BLACK)
        wn = board.pieces(chess.KNIGHT, chess.WHITE)
        bn = board.pieces(chess.KNIGHT, chess.BLACK)
        wb = board.pieces(chess.BISHOP, chess.WHITE)
        bb = board.pieces(chess.BISHOP, chess.BLACK)
        wr = board.pieces(chess.ROOK, chess.WHITE)
        br = board.pieces(chess.ROOK, chess.BLACK)
        wq = board.pieces(chess.QUEEN, chess.WHITE)
        bq = board.pieces(chess.QUEEN, chess.BLACK)
        wk = board.pieces(chess.KING, chess.WHITE)
        bk = board.pieces(chess.KING, chess.BLACK)

        pawnsq = sum([pawntable[i] for i in wp])
        pawnsq= pawnsq + sum([-pawntable[chess.square_mirror(i)]
                                        for i in bp])
        knightsq = sum([knightstable[i] for i in wn])
        knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)]
                                        for i in bn])
        bishopsq= sum([bishopstable[i] for i in wb])
        bishopsq= bishopsq + sum([-bishopstable[chess.square_mirror(i)]
                                        for i in bb])
        rooksq = sum([rookstable[i] for i in wr])
        rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)]
                                        for i in br])
        queensq = sum([queenstable[i] for i in wq])
        queensq = queensq + sum([-queenstable[chess.square_mirror(i)]
                                        for i in bq])
        kingsq = sum([kingstable[i] for i in wk])
        kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)]
                                        for i in bk])

        return pawnsq + knightsq + bishopsq+ rooksq+ queensq + kingsq 


class Engine():
    def __init__(self, depth):
        self.depth = depth
        self.board = chess.Board()
        self.evaluator = Evaluator()
        self.search = search.negamax_alpha_beta
        self.transition_table = {}

    def make_move(self, uci):
        move = chess.Move.from_uci(uci)
        self.board.push(move)

    def find_best_move(self):
        turn = 1 if self.board.turn else -1
        print('current turn', turn, self.board.turn)
        print(self.board)
        #score, move = self.search(self.board, self.evaluator, self.depth, turn)
        score, move, depth= self.search(self, self.depth, -9999, 9999, turn, root=True,) 
        return move, score
    
