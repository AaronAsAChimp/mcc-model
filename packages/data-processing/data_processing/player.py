import asyncio
from datetime import UTC, datetime
import logging
import os
import re
from typing import Optional
import uuid

import aiohttp
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from shared_core.data import MccIslandStats, PlayerData, PlayerMapping

logger = logging.getLogger(__name__)

MOJANG_MAX_BATCH_SIZE = 10

MEDIA_WIKI_API_URL = "https://mcchampionship.fandom.com/api.php"
# MediaWiki's `prop=revisions` accepts up to 50 titles per request.
MEDIA_WIKI_MAX_BATCH_SIZE = 50

# Minecraft account names only use [A-Za-z0-9_]. The wiki decorates some
# ign values with notes and former names, so validate before saving.
IGN_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,16}$")
IGN_LINE_PATTERN = re.compile(r"^\|\s*ign\s*=\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def clean_wikitext_ign(raw_ign: str) -> Optional[str]:
	"""Extract a Minecraft account name from the raw `ign` template value.

	The wiki decorates some values with event notes, former names and
	leftover template markup, e.g. `AvidMc_<br><small>(formerly AvidYt)</small>`.
	Only the most recent plain name is kept.
	"""

	# Strip editorial comments left in the wikitext.
	value = re.sub(r"<!--.*?-->", "", raw_ign, flags=re.DOTALL)

	# When a player changed their IGN, each name is separated by a <br>
	# with the current name last, so walk backwards until a segment
	# survives cleaning.
	for segment in reversed(re.split(r"<br\s*/?>", value)):
		segment = re.sub(r"<small>.*?</small>", "", segment, flags=re.DOTALL | re.IGNORECASE)
		segment = re.sub(r"\{\{[^{}]*\}\}", "", segment)
		# The value is sometimes the tail of an inline template call so
		# stray closing braces can be left behind.
		segment = segment.strip().rstrip("}").strip()

		if segment:
			return segment

	return None


async def populate_player_igns(player_data: PlayerMapping):
	"""Populate each player's `ign` from their MCC wiki page.

	The wiki only fills in the ign field when it differs from the player's
	wiki name, so players without one are left untouched.
	"""

	needs_ign = [
		name
		for [name, player] in player_data.items()
		if name != '$schema' and player.get('ign') is None
	]

	if len(needs_ign) == 0:
		return

	print(f'Fetching IGNs for { len(needs_ign) } players')

	async with aiohttp.ClientSession() as session:
		for i in range(0, len(needs_ign), MEDIA_WIKI_MAX_BATCH_SIZE):
			batch = needs_ign[i:i + MEDIA_WIKI_MAX_BATCH_SIZE]

			params = {
				'action': 'query',
				'prop': 'revisions',
				'rvprop': 'content',
				'titles': '|'.join(batch),
				'redirects': '1',
				'format': 'json',
			}

			async with session.get(MEDIA_WIKI_API_URL, params=params) as res:
				data = await res.json()

			# MediaWiki rewrites titles (case normalization and #REDIRECT)
			# so follow those chains to find each player's page.
			query = data.get('query', {})
			pages = {page['title']: page for page in query.get('pages', {}).values()}
			normalized = {n['from']: n['to'] for n in query.get('normalized', [])}
			redirects = {r['from']: r['to'] for r in query.get('redirects', [])}

			for name in batch:
				title = normalized.get(name, name)
				title = redirects.get(title, title)
				page = pages.get(title)

				if page is None or 'missing' in page:
					logger.warning("No wiki page for %s", name)
					continue

				content = page.get('revisions', [{}])[0].get('*', '')
				match = IGN_LINE_PATTERN.search(content)

				if match is not None:
					ign = clean_wikitext_ign(match.group(1))

					if ign is None or not IGN_PATTERN.match(ign):
						logger.warning("Unparsable ign for %s: %r", name, ign)
						continue
				else:
					# If no ign name on the page, treat their name as the ign
					ign = name

				player_data[name]['ign'] = ign

			# Be polite to the wiki between batches.
			await asyncio.sleep(1)

async def populate_player_uuids(player_data: PlayerMapping):
	ign_mapping: dict[str, PlayerData] = {}

	for [name, player] in player_data.items():
		if name == '$schema':
			continue

		uuid = player.get('uuid')
		ign = player.get('ign')

		if uuid is None and ign is not None:
			# Minecraft names are case-insensitive, and the lookup returns
			# the account's canonical casing.
			ign_mapping[ign.casefold()] = player

	ids = list(ign_mapping.keys())
	unresolved_ids = set(ids)

	print('Populating UUIDs for:', ', '.join(ids))

	for i in range(0, len(ids), MOJANG_MAX_BATCH_SIZE):
		batch = ids[i:i + MOJANG_MAX_BATCH_SIZE]

		async with aiohttp.ClientSession() as session:
			async with session.post('https://api.mojang.com/minecraft/profile/lookup/bulk/byname', json=batch) as res:
				results = await res.json()

				for result in results:
					normalized_ign = result['name'].casefold()
					player = ign_mapping[normalized_ign]
					unresolved_ids.remove(normalized_ign)

					# Use the account's canonical casing for the ign.
					player['ign'] = result['name']
					player['uuid'] = result['id']

	for name in unresolved_ids:
		normalized_ign = name.casefold()
		player = ign_mapping[normalized_ign]
		player['uuid'] = "unresolvable"

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