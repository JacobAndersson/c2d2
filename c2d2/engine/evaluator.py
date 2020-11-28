import torch
import chess
from c2d2.train.models.simpleModel import SimpleModel
from c2d2.train.models.smaller import Smaller
from c2d2.train.models.quant import Quant
from c2d2.train.modelInput import getBoardRep, handCraftedFeatures

class Evaluator():
    def __init__(self):
        self.model = Quant()
        self.model.load_state_dict(torch.load('./c2d2/train/checkpoints/model_smaller_epoch_14.pt'))
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    def eval(self, board):
        brd = torch.Tensor(getBoardRep(board)).unsqueeze(0)
        features = torch.Tensor(handCraftedFeatures(board)).unsqueeze(0)
        ev = self.model(brd, features).item()
        return ev 
 
def start():
    board = chess.Board()
    evaluator = Evaluator()
    ev = evaluator.eval(board)
