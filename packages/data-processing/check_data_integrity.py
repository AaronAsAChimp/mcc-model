import json

from shared_core import ExtractsFile
from shared_core.data import PlayerDataFile

all_players = set()

with ExtractsFile("data/extracts.jsonl") as iter:
    for event in iter:
        player_coins = event.get('player_coins')
        teams = event.get('teams')
        total_team_coins = 0
        total_player_coins = 0

        players_from_coins = set(player_coins)
        players_from_teams = set()

        for team in teams.values():
            players_from_teams = players_from_teams.union(team.get('players'))
            total_team_coins += team['coins']

        for player_coins in event['player_coins'].values():
            total_player_coins += player_coins

        if total_player_coins > total_team_coins:
            print(f'Error in coins for {event['name']},\nPlayer coins are un-multiplied so they should always be less than the team coins.')
            print('Total team coins:', total_team_coins)
            print('Total player coins:', total_player_coins)

        all_players = all_players.union(players_from_teams)

        missing_in_coins = players_from_teams.difference(players_from_coins)
        missing_in_teams = players_from_coins.difference(players_from_teams)

        # find missing players

        if len(missing_in_coins) != 0 and len(missing_in_teams) != 0:
            print('---')
            print(event.get('wiki_url'))
            print('Not in coins:')
            print(missing_in_coins)
            print('Not in teams:')
            print(missing_in_teams)


with PlayerDataFile("data/players.json") as player_data:
    all_players = {player_data.normalize_name(player) for player in all_players}

    with open("data/players.json") as players_json:
        no_data = all_players.difference(json.load(players_json).keys())

        if len(no_data) > 0:
            print('No player data:', sorted(list(no_data), key=str.lower))
