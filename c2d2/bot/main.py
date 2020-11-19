import berserk

from c2d2.bot.game_client import Game

session = berserk.TokenSession(token)
client = berserk.Client(session)

acceptChallenge = True

for event in client.bots.stream_incoming_events():
    print(event)

    if event['type'] == 'challenge':
        game_id = event['challenge']['id']

        if acceptChallenge:
            client.bots.accept_challenge(game_id)
            acceptChallenge = False
        else:
            client.bots.decline_challenge(game_id)

    elif event['type'] == 'gameStart':
        print(event)
        game_id = event['game']['id']
        game = Game(client, game_id)
        game.run()



    

