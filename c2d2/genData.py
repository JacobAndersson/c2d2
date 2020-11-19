import glob
import random
import threading
import queue

import chess.pgn
import chess.engine

DEPTHS = [0,3];
SAMPLING_RATE = 0.33

PATHS = queue.Queue()
FILES = glob.glob('./c2d2/data/games/*.pgn')

for f in FILES:
    PATHS.put(f)

BOARDS = queue.Queue()

SAVEFILE = open('./c2d2/dataset_zero.csv', 'w')
SAVEFILE.write('fen,depth0,depth3\n')

WRITE_LOCK = threading.Lock()
WRITE_BUFFER = queue.Queue()

def saveBoard(board, scores):
    currentFen = board.fen()
    columns = [currentFen] + scores
    currentRow = ','.join(columns) + '\n'

    if not WRITE_LOCK.locked():
        WRITE_LOCK.acquire()
        current_buff = list(WRITE_BUFFER.queue)
        WRITE_BUFFER.queue.clear() 
        
        current_buff.append(currentRow)

        for row in current_buff:
            SAVEFILE.write(row)

        WRITE_LOCK.release()
    else:
        WRITE_BUFFER.put(currentRow)


def stockfish_evaluation(board, depth, engine):
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    try:
        score = int(str(result['score'].white()))
        return score
    except ValueError:
        return None


def evalBoard(board, engine):
    scores = []
    for depth in DEPTHS:
        currentScore = stockfish_evaluation(board, depth, engine)
        if currentScore is not None:
            scores.append(str(currentScore))
        else:
            scores.append('0')
    return scores

def finishGame(board, engine):
    while not board.is_game_over():
        if random.random() < SAMPLING_RATE:
            scores = evalBoard(board, engine) 
            saveBoard(board, scores)

        result = engine.play(board, chess.engine.Limit(depth=1))
        board.push(result.move)

def playFile(path, engine):
    pgn = open(path, 'r')
    game = chess.pgn.read_game(pgn)
    board = game.board()

    doneMoves = 0
    for move in game.mainline_moves():
        doneMoves += 1
        board.push(move)
        if random.random() < SAMPLING_RATE and doneMoves > 6:
            scores = evalBoard(board, engine)
            saveBoard(board, scores)
    
    if (doneMoves < 6):
        return;

    if not board.is_game_over():
        variants = []
        doneMoves = 0
        allMoves = [move for move in board.legal_moves]

        if len(allMoves) > 5:
            allMoves  = random.sample(allMoves, 5)

        for move in allMoves:
            board.push(move)
            BOARDS.put(board.copy())
            board.pop()


def findMoves():
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")

    while not PATHS.empty(): 
        if not BOARDS.empty():
            brd = BOARDS.get()
            finishGame(brd, engine)
        else:
            pth = PATHS.get()
            print('num: {}, path: {}'.format(5449091 - PATHS.qsize(), pth))
            playFile(pth, engine)

    engine.close()

def start():
    threads = []
    for i in range(6):
        t = threading.Thread(target=findMoves)
        threads.append(t)
        t.start()

    [x.join() for x in threads]

    SAVEFILE.close()

if __name__ == '__main__':
    start()
