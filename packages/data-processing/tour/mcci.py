import asyncio
from datetime import datetime, UTC
import os
import uuid

from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from shared_core.data import MccIslandStats, PlayerMapping


async def populate_mcc_island_data(player_data: PlayerMapping):
	api_key = os.getenv("NOXCREW_API_KEY")

	if api_key is None:
		raise Exception("Noxcrew API key is required")

	update_timestamp = datetime.now(UTC).isoformat()
	players = list(player_data.items())
	transport = AIOHTTPTransport(
		url="https://api.mccisland.net/graphql",
		headers={
			"X-API-Key": api_key
		}
	)

	client = Client(transport=transport)

	async with client as session:
		for [name, player] in players:
			if name == '$schema':
				continue

			if player.get('island') is not None:
				continue

			player_hex_id = player.get("uuid")

			if player_hex_id is not None and player_hex_id != "unresolvable":
				query = gql("""
					query GetPlayerStats($uuid: UUID!) {
					  player(uuid: $uuid) {
					    uuid
					    username
					    statistics {
					      battleBoxRoundsPlayed: rotationValue(statisticKey: "battle_box_rounds_played", rotation: LIFETIME)
					      battleBoxRoundsWon: rotationValue(statisticKey: "battle_box_team_rounds_won", rotation: LIFETIME)
					      tgttosGames: rotationValue(statisticKey: "tgttos_games_played", rotation: LIFETIME)
					      tgttosTop8: rotationValue(statisticKey: "tgttos_top_eight", rotation: LIFETIME)
					      tgttosWon: rotationValue(statisticKey: "tgttos_first_place", rotation: LIFETIME)
					      skyBattleGames: rotationValue(statisticKey: "sky_battle_games_played", rotation: LIFETIME)
					      skyBattleTop3: rotationValue(statisticKey: "sky_battle_survival_top_three", rotation: LIFETIME)
					      skyBattleWon: rotationValue(statisticKey: "sky_battle_survival_first_place", rotation: LIFETIME)
					    }
					  }
					}
				""")

				player_uuid = str(uuid.UUID(player_hex_id))

				query.variable_values = {
					"uuid": player_uuid
				}

				# Execute the query
				result = await session.execute(query)
				stats = result.get('player', {}).get('statistics')

				island_stats: MccIslandStats = {
					"lastupdate": update_timestamp,
				}

				if stats is not None:
					battlebox_games = stats.get("battleBoxRoundsPlayed")
					battlebox_wins = stats.get("battleBoxRoundsWon")

					if battlebox_games is not None or battlebox_wins is not None:
						island_stats['battlebox'] = {
							"games": battlebox_games,
							"wins": battlebox_wins
						}

					skybattle_games = stats.get("skyBattleGames")
					skybattle_wins = stats.get("skyBattleWon")

					if skybattle_games is not None or skybattle_wins is not None:
						island_stats['skybattle'] = {
							"games": skybattle_games,
							"wins": skybattle_wins
						}

					tgttos_games = stats.get("tgttosGames")
					tgttos_wins = stats.get("tgttosWon")

					if tgttos_games is not None or tgttos_wins is not None:
						island_stats['tgttos'] = {
							"games": tgttos_games,
							"wins": tgttos_wins
						}

				player['island'] = island_stats

			await asyncio.sleep(2)
