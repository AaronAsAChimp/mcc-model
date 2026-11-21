import asyncio
import logging
from os.path import exists
import re
import random

from playwright.async_api import Page, async_playwright
from shared_core.data import PandoraPlayerData, PlayerData, PlayerDataFile, PlayerMapping

logger = logging.getLogger(__name__)


PLAYERS_PAGE = 'https://pandoras-box-event.fandom.com/wiki/Category:Players'

PLACE_AND_POINTS = re.compile(r'(\d+)(?:th|rd|st|nd)\([\d,]+\)', re.I)

PANDORA_PLAYERS_PER_TEAM = 4
PANDORA_TEAMS = 8
PANDORA_TOTAL_PLAYERS = PANDORA_PLAYERS_PER_TEAM * PANDORA_TEAMS

async def pandora_get_players(page: Page) -> list[str]:
	players = page.locator('.category-page__member-link')

	links = await players.evaluate_all('links => links.map(link => link.href)')

	return links


async def pandora_get_player_data_team_events(page: Page) -> list[list[str]] | None:
	"""
	Get the player data when solo events are *not* present.
	"""

	heading = page.locator('h2').filter(has=page.get_by_text("Event History"))
	rows = heading.locator("+ table tr")

	# count = await rows.count()

	# logger.debug(f"Tables: { count }")

	results = await rows.evaluate_all('''(rows, min) => {
	const results = [];

	for (const row of rows) {
		const cells = Array.from(row.getElementsByTagName('td'));

		if (cells.length === 4) {
			results.push(cells.map(el => el.textContent))
		}
	}

	return results;
}''')

	if len(results) == 0:
		return None

	return results


async def pandora_get_player_data_solo_events(page: Page) -> list[list[str]] | None:
	"""
	Get the player data when solo events are present.
	"""
	heading = page.locator('h3').filter(has=page.get_by_text("Team Events"))
	rows = heading.locator("+ table tr")

	# count = await rows.count()

	# logger.debug(f"Tables: { count }")

	results = await rows.evaluate_all('''(rows, min) => {
	const results = [];

	for (const row of rows) {
		const cells = Array.from(row.getElementsByTagName('td'));

		if (cells.length === 4) {
			results.push(cells.map(el => el.textContent))
		}
	}

	return results;
}''')

	if len(results) == 0:
		return None

	return results


async def pandora_get_player_data(page: Page) -> PandoraPlayerData:
	# count = await rows.count()

	# logger.debug(f"Tables: { count }")

	results = await pandora_get_player_data_team_events(page)

	if results is None:
		results = await pandora_get_player_data_solo_events(page)

	if results is None:
		raise Exception('Could not find player data')

	player_data: PandoraPlayerData = {
		'total_events': 0,
		'events_won': 0,
		'team_ranking': [],
		'player_ranking': []
	}

	for row in results:
		_, team_rank_str, player_rank_str, won_str = row

		team_rank_match = PLACE_AND_POINTS.match(team_rank_str)

		player_data['total_events'] += 1

		if team_rank_match is not None:
			team_place = float(team_rank_match.group(1))

			# A player subbed out mid-event gets a place above the participant
			# count on the wiki. That is not a real placement and would fall
			# outside the [0, 1] ranking scale, so skip it.
			if team_place <= PANDORA_TEAMS:
				player_data['team_ranking'].append(1 - (team_place / PANDORA_TEAMS))

		player_rank_match = PLACE_AND_POINTS.match(player_rank_str)

		if player_rank_match is not None:
			player_place = float(player_rank_match.group(1))

			if player_place <= PANDORA_TOTAL_PLAYERS:
				player_data['player_ranking'].append(1 - (player_place / PANDORA_TOTAL_PLAYERS))

		won = 'won' in won_str.lower()

		if won:
			player_data['events_won'] += 1

	logger.debug(f'Player data: { player_data }')

	return player_data


async def pandora_load(player_data: PlayerMapping):
	queue = []
	player_lookup = PlayerDataFile.from_player_mapping(player_data)

	if exists('./pandoras-queue.txt'):
		with open('./pandoras-queue.txt', 'r') as fd:
			queue = [item.strip() for item in  fd.readlines()]

	async with async_playwright() as playwright:
		firefox = playwright.firefox

		browser = await firefox.launch()
		page = await browser.new_page()

		player_page_links = {}

		if len(queue) == 0:
			await page.goto(PLAYERS_PAGE, wait_until="domcontentloaded", timeout=30_000)

			await page.wait_for_selector('#firstHeading', timeout=5_000)

			queue = await pandora_get_players(page)

			with open('./pandoras-queue.txt', 'w') as fd:
				fd.writelines(item +'\n' for item in queue)

		for link in queue:
			_, _, player_name = link.rpartition('/')

			if player_name.startswith('Category:'):
				continue

			player_name = player_lookup.normalize_name(player_name.replace('_', ' '))

			if player_name not in player_data:
				logger.debug('Missing player %s', player_name)
			else:
				if player_data[player_name].get('pandora') is not None:
					continue

			player_page_links[player_name] = link


		if len(player_page_links) == 0:
			logger.info("All players processed")
			return

		items = random.choices(list(player_page_links.items()), k=1)


		for player_name, link in items:
			sleep_time = 10 + (random.random() * 10)
			await asyncio.sleep(sleep_time)

			_, _, player_name = link.rpartition('/')

			player_name = player_lookup.normalize_name(player_name.replace('_', ' '))

			player: PlayerData | None = player_data.get(player_name)

			if player is None:
				player = {}
				player_data[player_name] = player

			logger.debug('loading data for %s', player_name)

			await page.goto(link, wait_until="domcontentloaded", timeout=30_000)

			try:
				await page.wait_for_selector('#firstHeading', timeout=5_000)
			except Exception as e:
				# logger.debug('Page contents %s', await page.content())
				# logger.exception('Error loading page', exc_info=e)
				raise e

			pandora_data = await pandora_get_player_data(page)

			player['pandora'] = pandora_data

