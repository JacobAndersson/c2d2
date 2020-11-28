import os
import berserk
from c2d2.bot.game_client import Game

def start(depth):
    token = os.getenv('API_TOKEN')
    bot_id = os.getenv('BOT_ID')
    print(token, bot_id)

    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    acceptChallenge = True

    for event in client.bots.stream_incoming_events():
        print(event)
        if event['type'] == 'challenge':
            game_id = event['challenge']['id']
            challenge = event['challenge']

            if challenge['challenger']['id'] == bot_id:
                continue

            if acceptChallenge:
                client.bots.accept_challenge(game_id)
                acceptChallenge = False
            else:
                client.bots.decline_challenge(game_id)

        elif event['type'] == 'gameStart':
            game_id = event['game']['id']
            game = Game(client, game_id, bot_id, depth)
            game.run()
        elif event['type'] == 'gameFinish':
            acceptChallenge = True

