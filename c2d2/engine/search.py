from chess.polyglot import zobrist_hash
from collections import namedtuple
from enum import Enum

Entry = namedtuple('Entry', 'value depth flag move')
class EntryFlag(Enum):
    EXACT = 1
    LOWERBOUND = 2
    UPPERBOUND = 3


def nega_max(board, evaluator, depth, color):
    if depth == 0 or board.is_game_over():
        return [color*evaluator.eval(board), None]

    best = [-9999, None]
    for move in board.legal_moves:
        board.push(move)
        (score, _) = nega_max(board, evaluator, depth - 1, -color)
        score = -score
        board.pop()

        if score > best[0]:
            best = [score, move]

    return best

def negamax_alpha_beta(engine, depth, alpha, beta, color, root=False, prob=1.0, curr_depth=0):
    alphaorig = alpha
    board = engine.board
    evaluator = engine.evaluator
    transition_table = engine.transition_table
    moves = list(board.legal_moves)
    skip_cache = False

    if root:
        tmp = []
        for move in moves:
            board.push(move)
            if board.is_checkmate() or not board.is_game_over():
                tmp.append(move)
            board.pop()

        if len(tmp) < len(moves) and len(tmp) > 0:
            moves = tmp
            skip_cache = True
    
    h = zobrist_hash(board)
    # ttEntry lookup
    ttEntry = transition_table.get(h)
    if not skip_cache and ttEntry and ttEntry.depth >= depth:
        if ttEntry.flag == EntryFlag.EXACT:
            return (ttEntry.value, ttEntry.move, ttEntry.depth)
        elif ttEntry.flag == EntryFlag.LOWERBOUND:
            alpha = max(alpha, ttEntry.value)
        elif ttEntry.flag == EntryFlag.UPPERBOUND:
            beta = min(beta, ttEntry.value)

        if alpha >= beta:
            return (ttEntry.value, ttEntry.move, ttEntry.depth)

    #if prob < prob_threshold:
    if depth == 0 or board.is_checkmate() or len(moves) == 0:
        return [quiesce(board, evaluator, alpha, beta, color), None, curr_depth]

    _max = [-99999, None, curr_depth]
    #if True and prob > (prob_threshold):

    for move in moves:
        board.push(move)
        (score, ch, node_depth) = negamax_alpha_beta(engine, depth - 1, -beta, -alpha, -color, curr_depth=curr_depth+1)
        score = -score
        board.pop()
        if score > _max[0]:
            _max = [score, move, node_depth]
        alpha = max(alpha, _max[0])
        if alpha >= beta:
            break

    # Store ttEntry
    value = _max[0]
    if value <= alphaorig:
        flag = EntryFlag.UPPERBOUND
    elif value >= beta:
        flag = EntryFlag.LOWERBOUND
    else:
        flag = EntryFlag.EXACT
    ttEntry = Entry(value=value, depth=depth, flag=flag, move=_max[1])
    transition_table[h] = ttEntry
    return _max

def quiesce(board, evaluator, alpha, beta, color, depth=10):
    standpat = color * evaluator.eval(board)
    if depth == 0:
        return standpat
    if standpat >= beta:
        return beta
    if alpha < standpat:
        alpha = standpat
    
    for child in board.legal_moves:
        if not board.is_capture(child):
            continue
        board.push(child)
        score = -quiesce(board, evaluator, -beta, -alpha, -color, depth - 1)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha
