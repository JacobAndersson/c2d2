import torch

from c2d2.train.models.simpleModel import SimpleModel
from c2d2.train.modelInput import getBoardRep, handCraftedFeatures

class Evaluator():
    def __init__(self):
        self.model = SimpleModel()
        self.model.eval()

    def eval(self, board):
        brd = torch.Tensor(getBoardRep(board)).unsqueeze(0)
        features = torch.Tensor(handCraftedFeatures(board)).unsqueeze(0)
        ev = self.model(brd, features).item()
        return ev 
 
def start():
    import chess
    board = chess.Board()
    evaluator = Evaluator()

    ev = evaluator.eval(board)
    print(ev)
