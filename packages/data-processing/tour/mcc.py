import asyncio
import logging
import re
from typing import Optional

import aiohttp


from shared_core.data import PlayerData, PlayerMapping

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

	logger.debug(f'Fetching IGNs for { len(needs_ign) } players: { ', '.join(needs_ign) }')

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

	if len(ids) == 0:
		return

	unresolved_ids = set(ids)

	logger.debug(f'Populating UUIDs for: { ', '.join(ids) }')

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
