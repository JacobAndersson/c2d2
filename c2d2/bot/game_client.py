import threading
from c2d2.engine.engine import Engine
import random

class Game(threading.Thread):
    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.current_state = next(self.stream)
        self.engine = Engine()

    def run(self):
        for event in self.stream:
            print('class move', event)
            if event['type'] == 'gameState':
                self.handle_state_change(event)
            elif event['type'] == 'chatLine':
                self.handle_chat_line(event)
    
    def handle_state_change(self, game_state):
        all_moves = game_state['moves']
        self.recreate_boards(all_moves)
        
        move = self.find_move()
        self.client.bots.make_move(self.game_id, move)

    def find_move(self):
        move = random.choice(list(self.engine.board.legal_moves))
        self.engine.board.push(move)

        return move.uci()




    def recreate_boards(self, moves):
        self.engine.board.reset()

        for move in moves.split(' '):
            print(move)
            self.engine.make_move(move)


    def handle_chat_line(self, chat_line):
        pass
