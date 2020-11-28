import threading
from c2d2.engine.engine import Engine
import random
import chess

class Game(threading.Thread):
    def __init__(self, client, game_id, bot_id, depth, **kwargs):
        super().__init__(**kwargs)
        self.bot_id = bot_id
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.engine = Engine(depth=depth)

    def run(self):
        if self.current_state:
            try:
                is_white = self.current_state['white']['id'] == self.bot_id
            except KeyError:
                is_white = False

            self.engine.color = chess.WHITE if is_white else chess.BLACK
            state = self.current_state['state']
            self.handle_state_change(state)


        for event in self.stream:
            if 'winner' in event:
                break

            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)
    
    def handle_state_change(self, game_state):
        all_moves = game_state['moves']
        self.recreate_boards(all_moves)
        
        if (self.engine.board.turn == self.engine.color):
            move, ev = self.find_move()
            print(move, ev)
            self.client.bots.make_move(self.game_id, move)

    def find_move(self):
        #move = random.choice(list(self.engine.board.legal_moves))
        move, evaluation  = self.engine.find_best_move()
        print(move, evaluation)
        self.engine.board.push(move)
        return move.uci(), evaluation

    def recreate_boards(self, moves):
        self.engine.board.reset()

        if moves == '':
            return None
        for move in moves.split(' '):
            self.engine.make_move(move)


    def handle_chat_line(self, chat_line):
        pass
