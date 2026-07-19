from shared_core import ExtractsFile

with ExtractsFile("data/extracts.jsonl") as iter:
    for event in iter:
        player_coins = event.get('player_coins')
        teams = event.get('teams')

        players_from_coins = set(player_coins)
        players_from_teams = set()

        for team in teams.values():
            players_from_teams = players_from_teams.union(team.get('players'))

        missing_in_coins = players_from_teams.difference(players_from_coins)
        missing_in_teams = players_from_coins.difference(players_from_teams)

        if len(missing_in_coins) != 0 and len(missing_in_teams) != 0:
            print('---')
            print(event.get('wiki_url'))
            print('Not in coins:')
            print(missing_in_coins)
            print('Not in teams:')
            print(missing_in_teams)

