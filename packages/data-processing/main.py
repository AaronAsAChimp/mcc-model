#!/usr/bin/env python
import asyncio
from datetime import date, datetime, timedelta
from enum import StrEnum, auto
import os
import time
from typing import Optional, TextIO
from playwright.async_api import Page, async_playwright
from dataclasses import dataclass, asdict
import logging
import json

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

MAX_WORKERS = 2
EVENTS_CATEGORY = "https://mcchampionship.fandom.com/wiki/Category:Events"

IGNORED_PAGES = set(
    [
        # Not actually an event
        "https://mcchampionship.fandom.com/wiki/MC_Championship",
        # Too scuffed to provide any useful data
        "https://mcchampionship.fandom.com/wiki/MC_Championship_Scuffed",
    ]
)

PLAYER_ALIASES = {
    "5uppps": "5up",
    "_M4NG0_": "M4NG0",
    "_Spelda_": "Spelda",
    "Alex_the_Rambler": "AlexTheRambler",
    "Amadeus_Fink": "Tom Hazell",
    "Antoneemc": "Anthony Toczek",
    "AQUINOby02": "AQUINO",
    "ashswagg": "ashswag",
    "AvaKrisTyson": "Ava Tyson",
    "AvidMc_": "AvidMc",
    "BearsCancelle": "BearsCancelled",
    "Bedgar_": "Bedgar",
    "bigbst4tz2": "Bigbst4tz2",
    "BigLyds": "SquidGaming",
    "billzo_": "billzo",
    "Block_Facts": "Block Facts",
    "blowstoise": "blaustoise",
    "bouphe": "Bouphe",
    "boxbox22": "BoxBox",
    "brionykay": "Brionykay",
    "BrryHrry": "Hrry",
    "Calvin": "Cxlvxn",
    "CapitanGatoYT": "Capitan Gato",
    "Caramello_sb": "Caramello",
    "ChilledChaosYT": "ChilledChaos",
    "ChippyGaming_": "ChippyGaming",
    "chunkpurse": "Chunk Purse",
    "CraigMeUp": "Craig Frost",
    "CydoniaPM": "Cydonia",
    "DeusAmir": "deusamir",
    "Drgluon": "DrGluon",
    "drsimonclark": "Dr. Simon Clark",
    "DuncanMassink": "Duncan Massink",
    "Eskay22": "Eskay",
    "Etho": "EthosLab",
    "falsesymmetry": "FalseSymmetry",
    "Fiona_McGoats": "Fionn Riches",
    "Fit": "FitMC",
    "Flabaliki": "James Turner",
    "FoolishG": "Foolish Gamers",
    "GamerBurry": "Strawburry17",
    "GeeNelly": "Gee Nelly",
    "GeeNelly": "Gee Nelly",
    "Geestargames": "Geestar",
    "GoodTimeWithScar": "GoodTimesWithScar",
    "graaavel": "Graaavel",
    "Guggleino": "Guggle",
    "hannahhkatee": "HannahKate",
    "heyGraecie": "Graecie",
    "Honeydew": "Simon Lane",
    "hydraticc": "Hydraticc",
    "IJevin": "iJevin",
    "infinitedrift": "InfiniteDrift",
    "Its_Dev_": "Dev",
    "ItsFundy": "Fundy",
    "ItsValkyrae": "Valkyrae",
    "JackSucksAtMC": "JackSucksAtLife",
    "JamalMC_": "JamalMC",
    "James_Prime": "Jameskii",
    "JamesDoesMining": "James Marriott",
    "JamesTurner": "James Turner",
    "jcharles": "James Charles",
    "JCtheCaster": "JC The Caster",
    "JeremyFrieser": "Jeremy Frieser",
    "Jestan": "Jestanii",
    "JHorseMC": "JHorse",
    "joehillssays": "Joe Hills",
    "JOEYGRACEFFA": "Joey Graceffa",
    "jvckbears": "jvckk",
    "KaptainSparklez": "CaptainSparklez",
    "KaraCorvus": "Kara Corvus",
    "KarlJacobs": "Karl Jacobs",
    "katherineeliz": "Katherine Elizabeth",
    "Keralis1": "Keralis",
    "Kierrrr": "Kier",
    "KKatamina_": "kkatamina_",
    "Kreekcraft": "KreekCraft",
    "KRISTlNA": "Kristina",
    "Krtzy": "Kratzy",
    "Krtzyy": "Kratzy",
    "Kryguyrocky": "Ryguyrocky",
    "KryticZeuZ": "KryticZeuz",
    "laurenzside": "LaurenzSide",
    "LazarCodeLazar": "LazarBeam",
    "LilyCraftChu": "LilyPichu",
    "LividCoffee": "Duncan Jones",
    "ludwigahgren": "Ludwig",
    "Magistrex": "RTGame",
    "marbar_x": "Marbar_x",
    "Markisparki66": "Marki Sparki",
    "MaxTLL": "MaxGGs",
    "MiniMuka": "Mini Muka",
    "misfitclovers": "misfitclover",
    "mizzrowe": "Mizzrowe",
    "Mousie_Mouse": "Mousie",
    "Mumbo": "Mumbo Jumbo",
    "musicman1017": "Musicman1017",
    "MysterySalesman": "Scot Griswold",
    "mysticat_": "Mysticat",
    "MythicalSausage": "Mythical Sausage",
    "NanoSounds": "NanoKim",
    "Nicacolas": "Nicacola",
    "NinaSerena": "Nina Serena",
    "NomadWorld": "NomadOfTheWorld",
    "noooreax": "nooreax",
    "Not_iiTzTimmy": "iiTzTimmy",
    "Nukerii": "Nukeri",
    "OrionSound": "TheOrionSound",
    "osiefish": "Osiefish",
    "Owenge_Juice": "OwengeJuice",
    "OX_Andy": "Farrantula",
    "OX_Ellen": "Ellen Rose",
    "OX_Jane": "Jane Douglas",
    "OX_Mike": "Mike Channell",
    "PaintingRainbows": "Rainbows",
    "PedguinTV": "Pedguin",
    "piso4": "Piso",
    "plumbella": "Plumbella",
    "QuirkyFace": "Joe Hickson",
    "Raebie": "RAEHASRABIES",
    "RainhoeVT": "Rainhoe",
    "Ravsu": "Ravs",
    "RedVelvetCake": "VelvetIsCake",
    "rendog": "Rendog",
    "Renthedog": "Rendog",
    "rhiga": "Ryan Higa",
    "RichardThornton": "Richard Thornton",
    "RuByc0": "Rubyco",
    "ryguyrocky": "Ryguyrocky",
    "SandwichLord_": "SandwichLord",
    "sarahsera": "Sarahsera",
    "septicbooper": "Jacksepticeye",
    "Shadoune777": "Shadoune666",
    "shadowatno0n": "shadowatnoon",
    "SherlockHulmes": "Mara Holmes",
    "ShubbleYT": "Shubble",
    "Sinnabuns4": "Sinnabuns",
    "SkyyartMC": "Skyyart",
    "sleepai": "sleepy",
    "Smallant1": "SmallAnt",
    "Spifeey": "Spifey",
    "steph0sims": "Steph0sims",
    "SteveCP": "Checkpoint",
    "stevesuptic": "Steven Suptic",
    "Stevesuptic_": "Steven Suptic",
    "sylvee_": "Sylvee",
    "TaneeshaHogan": "Taneesha",
    "Tango": "TangoTek",
    "TBNRfrags": "PrestonPlayz",
    "TeanaKitten": "TinaKitten",
    "The_Eret": "Eret",
    "TheFabo": "Fabo",
    "TheRealSquiddo": "Squiddo",
    "tibbycaps": "TibbyCaps",
    "tomatobisque_": "TomatoBisqueSoup",
    "Tubbo_": "Tubbo",
    "Tubbo_Clone_": "Tubbo",
    "Vikkkkkstarrrrrr": "Vikkstar123",
    "vintageapples": "vintage_applesauce",
    "VoiceoverPeteAce": "VoiceoverPete",
    "Weeannon": "Rhiannon",
    "WilburSoot": "Wilbur Soot",
    "WithMay": "WithMayX",
    "Wondernat_": "Wondernat",
    "Xephos": "Lewis Brindley",
    "Xisuma": "Xisumavoid",
    "xRizzBars": "xChocoBars",
    "yammy_xox": "Yammyxox",
    "Zoeya": "Zoey Proasheck",
    "Zylush": "Zylus",
}


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class MccTeamName(StrEnum):
    YellowYaks = auto()
    AquaAxolotls = auto()
    BlueBats = auto()
    CyanCoyotes = auto()
    GreenGeckos = auto()
    LimeLlamas = auto()
    OrangeOcelots = auto()
    PinkParrots = auto()
    PurplePandas = auto()
    RedRabbits = auto()

    @classmethod
    def from_str(cls, val: str) -> Optional[MccTeamName]:
        team = None

        match val:
            case "Red Rabbits" | "Red Ravens" | "Red Reindeer" | "Krimson Krakens" | "Simon's Angels":
                team = cls.RedRabbits
            case "Orange Ocelots" | "Orange Oozes" | "Coral Carollers" | "Ginger Breadmen" | "What the Flux?!":
                team = cls.OrangeOcelots
            case "Yellow Yaks" | "Yellow Yolks" | "Yellow Chickens" | "Mustard Mummies" | "Yellow Yetis" | "Rick Astley's Shorts":
                team = cls.YellowYaks
            case "Lime Llamas" | "Lime Liches" | "Mint Mistletoes" | "Dodgy Doodlers":
                team = cls.LimeLlamas
            case "Green Geckos" | "Green Guardians" | "Green Goblins" | "Emerald Elves" | "The Goblins":
                team = cls.GreenGeckos
            case "Cyan Coyotes" | "Cyan Creepers" | "Cyan Centipedes" | "Cyan Candy Canes" | "Cerulean Candy Canes" | "Not the Red Team":
                team = cls.CyanCoyotes
            case "Aqua Axolotls" | "Aqua Horses" | "Aqua Abominations" | "Teal Turkeys" | "G.O.A.T":
                team = cls.AquaAxolotls
            case "Blue Bats" | "Sapphire Simmers" | "Blue Black Cats" | "Sapphire Santas" | "Blue Banshees" | "Power Bottoms":
                team = cls.BlueBats
            case "Purple Pandas" | "Violet Vampires" | "Purple Penguins" | "BEEEEEEEES":
                team = cls.PurplePandas
            case "Pink Parrots" | "Fuchsia Frankensteins" | "Pink Presents" | "Pink Piglets" | "UH OH CHONKY":
                team = cls.PinkParrots

        return team


@dataclass
class MccTeam:
    placement: int
    players: list[str]
    coins: int = 0


class MccGame(StrEnum):
    # Current Games
    AceRace = auto()
    BattleBox = auto()
    BuildMart = auto()
    BingoButFast = auto()
    GridRunners = auto()
    HoleInTheWall = auto()
    Meltdown = auto()
    ParkourTag = auto()
    ParkourWarrior = auto()
    RailroadRush = auto()
    RocketSpleefRush = auto()
    SandsOfTime = auto()
    SkyBattle = auto()
    SurvivalGames = auto()
    Tgttosawaf = auto()
    Dodgebolt = auto()

    # Retired
    FootRace = auto()
    LockoutBingo = auto()
    ParkourWarriorOrig = auto()
    RocketSpleef = auto()
    Skyblockle = auto()

    # Special Event
    ParkourWarriorSurvivor = auto()
    BuildMartIrl = auto()

    @classmethod
    def from_str(cls, val: str) -> Optional[MccGame]:
        game = None

        match val:
            case "Ace Race":
                game = cls.AceRace
            case "Battle Box":
                game = cls.BattleBox
            case "Big Sales at Build Mart":
                game = cls.BuildMart
            case "Bingo but Fast":
                game = cls.BingoButFast
            case "Grid Runners":
                game = cls.GridRunners
            case "Hole in the Wall":
                game = cls.HoleInTheWall
            case "Meltdown":
                game = cls.Meltdown
            case "Parkour Tag":
                game = cls.ParkourTag
            case "Parkour Warrior":
                game = cls.ParkourWarrior
            case "Railroad Rush":
                game = cls.RailroadRush
            case "Rocket Spleef Rush":
                game = cls.RocketSpleefRush
            case "Sands of Time":
                game = cls.SandsOfTime
            case "Sky Battle":
                game = cls.SkyBattle
            case "Survival Games":
                game = cls.SurvivalGames
            case "TGTTOSAWAF":
                game = cls.Tgttosawaf
            case "Dodgebolt":
                game = cls.Dodgebolt

            case "Foot Race":
                game = cls.FootRace
            case "Lockout Bingo":
                game = cls.LockoutBingo
            case "Parkour Warrior (Original)":
                game = cls.ParkourWarriorOrig
            case "Rocket Spleef":
                game = cls.RocketSpleef
            case "Skyblockle":
                game = cls.Skyblockle

            case "Parkour Warrior Survivor":
                game = cls.ParkourWarriorSurvivor
            case "Build Mart IRL":
                game = cls.BuildMartIrl

        return game


class MccPlayer:
    minecraft_handle: str
    first_mcc: Optional[date]
    number_of_mccs: int

    def __init__(self, minecraft_handle: str) -> None:
        self.minecraft_handle = minecraft_handle
        self.first_mcc = None
        self.number_of_mccs = 0


@dataclass
class MccEvent:
    name: str
    wiki_url: str
    date: date
    teams: dict[MccTeamName, MccTeam]
    winning_team: MccTeamName
    second_place_team: MccTeamName
    player_coins: dict[str, int]
    games: list[MccGame]

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date == value.date

    def __ne__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date != value.date

    def __lt__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date < value.date

    def __gt__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date > value.date

    def __le__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date <= value.date

    def __ge__(self, value: object) -> bool:
        if not isinstance(value, MccEvent):
            return NotImplemented

        return self.date >= value.date


class Results:
    successes: int
    failures: list[str]

    def __init__(self):
        self.successes = 0
        self.failures = []


async def extract_name(page: Page) -> Optional[str]:
    headings = await page.locator(f"h1").evaluate_all(
        "els => els.map(el => el.textContent)"
    )

    if len(headings) > 0:
        return headings[0].strip()

    return None


async def extract_sidebar_date(page: Page, source: str) -> Optional[date]:
    dates = await page.locator(f'[data-source="{source}"]').evaluate_all(
        'els => els.map(el => el.querySelector(".pi-data-value")?.textContent)'
    )

    if len(dates) > 0:
        return date.strptime(dates[0], "%B %d, %Y")

    return None


async def extract_sidebar_team(page: Page, source: str) -> Optional[MccTeamName]:
    dates = await page.locator(f'[data-source="{source}"]').evaluate_all(
        'els => els.map(el => el.querySelector(".pi-data-value")?.textContent)'
    )

    if len(dates) > 0:
        return MccTeamName.from_str(dates[0].strip())

    return None


async def extract_teams(page: Page) -> dict[MccTeamName, MccTeam]:
    teams_table = await page.locator("#Teams").evaluate_all(
        """els => {
		const rows = els
			.map(el => {
				return Array.from(el.parentElement.nextElementSibling.getElementsByTagName('tbody'))
			})
			.flat(1)
			.map(el => {
				return Array.from(el.getElementsByTagName('tr'))
			})
			.flat(1)
			.map(el => {
				const teamImg = el.children[0].querySelector('img');
				const playersContainer = el.children[1].querySelector('.nomobile') || el.children[1];
				const rank = el.children[2].textContent;

				return [
					teamImg?.alt,
					playersContainer.textContent.split(/[•·]/).map(name => name.trim()).filter(Boolean),
					parseInt(rank.substring(0, rank.length - 2), 10),
				]
			});

		rows.shift()

		return rows;
	}"""
    )

    teams: dict[MccTeamName, MccTeam] = {}

    for [team_name, players, placement] in teams_table:
        team = MccTeamName.from_str(team_name)

        for i in range(len(players)):
            player = players[i]

            if player in PLAYER_ALIASES:
                players[i] = PLAYER_ALIASES[player]

        if len(players) != 4:
            logger.warning(
                "Wrong number of players (%s) for team %s", len(players), team_name
            )
            logger.debug("Page url: %s", page.url)

        if team is not None:
            teams[team] = MccTeam(placement, players)
        else:
            logger.error("No team for %s", team_name)

    return teams


async def extract_games(page: Page) -> list[MccGame]:
    games_headings = await page.locator("#Games").evaluate_all(
        """els => {
		const rows = els
			.map(el => {
				const headings = [];
				let nextSibling = el.parentElement.nextElementSibling;

				while (nextSibling !== null && nextSibling.tagName !== 'H2') {
					if (nextSibling.tagName ==='H3') {
						headings.push(nextSibling.querySelector('a').title)
					}

					nextSibling = nextSibling.nextElementSibling
				}

				return headings;
			})
			.flat(1);

		return rows;
	}"""
    )

    games: list[MccGame] = []

    for game_name in games_headings:
        game = MccGame.from_str(game_name)

        if game is not None:
            games.append(game)
        else:
            logger.error("No game for %s", game_name)

    return games


async def extract_team_coins(page: Page) -> dict[MccTeamName, int]:
    """Extract team coin totals from the Final Team Standings table."""

    coins_data = await page.locator("#Final_Team_Standings").evaluate_all(
        """els => {

			const rows = els.map(el => {
				return Array.from(el.parentElement.nextElementSibling.getElementsByTagName('tbody'));
			})
			.flat(1)
			.map(el => {
				return Array.from(el.getElementsByTagName('tr'))
			})
			.flat(1)
			.map((el) => {
				const teamImg = el.querySelector('img');

				// This is a header row
				if (!teamImg) {
					return [null, null];
				}

				const coinsCell = el.children[2];

				return [
					teamImg.alt,
					parseInt(coinsCell.textContent.replaceAll(',', ''))
				]
			});

			return rows;
		}"""
    )

    result: dict[MccTeamName, int] = {}

    if coins_data:
        for [team_name, coins] in coins_data:
            if team_name is None:
                continue

            team = MccTeamName.from_str(team_name)
            if team is not None:
                result[team] = coins
            else:
                logger.warning("Unknown team in Final Team Standings: %s", team_name)

    return result


async def extract_player_coins(page: Page) -> dict[str, int]:
    coins_data = await page.locator("#Final_Individual_Standings").evaluate_all(
        """els => {
        	const rows = els.map(el => {
        		return Array.from(el.parentElement.nextElementSibling.getElementsByTagName('tbody'));
        	})
        	.flat(1)
        	.map(el => {
        		return Array.from(el.getElementsByTagName('tr'))
        	})
        	.flat(1)
        	.slice(1) // Skip the table header

        	const player_coins = [];

        	let rowspan = 0;
        	let coins = 0;

        	for (const row of rows) {
        		// When there is a tie the table uses a rowspan so we have to
        		// index the columns from the right.
        		const cols = row.children.length;

        		let playerNameCell;

        		if (rowspan > 0) {
        			playerNameCell = row.children[cols - 1];

        			rowspan -= 1;

        		} else {
	        		const coinsCell = row.children[cols - 1];

	        		rowspan = +coinsCell.getAttribute('rowspan') - 1;
	        		coins = parseInt(coinsCell.textContent.replaceAll(',', ''));

	        		playerNameCell = row.children[cols - 2];
        		}

        		let playerName = playerNameCell.textContent.trim();

        		if (playerName.endsWith('*')) {
        			playerName = playerName.slice(0, playerName.length - 1);
        		}

        		player_coins.push([
        			playerName,
        			coins
        		]);

        	}

        	return player_coins;
        }"""
    )

    print(f"coins {coins_data}")

    result: dict[str, int] = {}

    if coins_data:
        for [player_name, coins] in coins_data:
            if player_name is None:
                print(f"No name {page.url}")
                continue

            if PLAYER_ALIASES.get(player_name) is not None:
                player_name = PLAYER_ALIASES[player_name]

            result[player_name] = coins

    return result


async def extract_mcc_event(
    logger: logging.Logger, page: Page, url: str
) -> Optional[MccEvent]:
    logger.debug("loading page")

    await page.goto(url, wait_until="domcontentloaded", timeout=30_000)

    await page.wait_for_selector('[data-source="date"]', timeout=1_000)

    [
        event_date,
        event_winner,
        event_second_place,
        name,
        teams,
        games,
        team_coins,
        player_coins,
    ] = await asyncio.gather(
        extract_sidebar_date(page, "date"),
        extract_sidebar_team(page, "winner"),
        extract_sidebar_team(page, "runner-up"),
        extract_name(page),
        extract_teams(page),
        extract_games(page),
        extract_team_coins(page),
        extract_player_coins(page),
    )

    if event_date is None:
        logger.warning("No event date, skipping %s", url)
        return None

    if event_winner is None:
        logger.warning("No event winner, skipping %s", url)
        return None

    if event_second_place is None:
        logger.warning("No event second place winner, skipping %s", url)
        return None

    logger.debug("Games: %s", games)

    for team_name, team_data in teams.items():
        team_data.coins = team_coins.get(team_name, 0)

    event = MccEvent(
        name=name if name is not None else "",
        wiki_url=url,
        date=event_date,
        teams=teams,
        winning_team=event_winner,
        second_place_team=event_second_place,
        player_coins=player_coins,
        games=games,
    )

    return event


async def worker(
    worker_id: int,
    queue: asyncio.Queue[str],
    results: Results,
    outfile: TextIO,
    page: Page,
):
    worker_logger = logger.getChild(f"Worker { worker_id }")

    print(f"Worker {worker_id} booted")
    while True:
        try:
            url = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        try:
            worker_logger.debug("Getting content")

            event = await extract_mcc_event(worker_logger, page, url)

            if event is not None:
                outfile.write(json.dumps(asdict(event), default=json_serial))

            results.successes += 1

            # worker_logger.debug('Event: %s', event)
        except Exception as e:
            results.failures.append(url)
            logger.exception("Exception processing URL: %s", url, exc_info=e)
        finally:
            worker_logger.debug("Task done")
            queue.task_done()


async def load_from_wiki():
    extracts_filename = "data/extracts.jsonl"

    already_extracted = set()

    if os.path.exists(extracts_filename):
        with open(extracts_filename, "r") as infile:
            for line in infile.readlines():
                extract_json = json.loads(line)
                wiki_url = extract_json.get("wiki_url")

                if wiki_url is not None:
                    already_extracted.add(wiki_url)

    async with async_playwright() as playwright:
        with open(extracts_filename, "a") as outfile:
            firefox = playwright.firefox

            browser = await firefox.launch()
            page = await browser.new_page()
            links = None

            try:
                print("Loading page...")

                await page.goto(
                    EVENTS_CATEGORY, wait_until="domcontentloaded", timeout=30_000
                )

                # print('Waiting for page...')

                # await page.wait_for_load_state('load')

                print("Getting links...")

                links = await page.locator("a").evaluate_all(
                    """
					els => {
						return els
							.filter(link => link.classList.contains('category-page__member-link'))
							.filter(link => {
								const href = link.href;

								return href.startsWith('https://mcchampionship.fandom.com/wiki/MC_Championship')
									|| href === 'https://mcchampionship.fandom.com/wiki/The_Ender_Cup'
									|| href === 'https://mcchampionship.fandom.com/wiki/Jingle_Jam_Championship';
							})
							.filter(link => !link.href.endsWith('(Series)'))
							.map(link => link.href)
					}
				"""
                )

                links = [
                    lnk
                    for lnk in links
                    if lnk not in IGNORED_PAGES and lnk not in already_extracted
                ][0:10]
                print(links)

                await page.close()

                # queue: asyncio.Queue[str] = asyncio.Queue()
                # tasks = []
                results: Results = Results()

                # for i in range(min(len(links), MAX_WORKERS)):
                # 	print(f'Booting worker {i}')
                # 	page = await browser.new_page()
                # 	task = asyncio.create_task(worker(i, queue, results, outfile, page))
                # 	tasks.append(task)

                # for link in links:
                # 	queue.put_nowait(link)

                started_at = time.monotonic()
                # await queue.join()

                for url in links:
                    page = await browser.new_page()

                    try:
                        logger.debug("Getting content")

                        event = await extract_mcc_event(logger, page, url)

                        if event is not None:
                            outfile.write(
                                json.dumps(asdict(event), default=json_serial)
                            )
                            outfile.write("\n")

                        results.successes += 1

                        # worker_logger.debug('Event: %s', event)
                    except Exception as e:
                        results.failures.append(url)
                        logger.exception(
                            "Exception processing URL: %s", url, exc_info=e
                        )
                    finally:
                        logger.debug("Task done")
                        await page.close()

                    time.sleep(10)

                total_slept_for = time.monotonic() - started_at

                # await asyncio.gather(*tasks)

                # print(f'{len(tasks)} workers took {total_slept_for:.2f} seconds to process {results.successes} successful results and { len(results.failures) } failures')
                print(
                    f"Took {total_slept_for:.2f} seconds to process {results.successes} successful results and { len(results.failures) } failures"
                )
                print("Failures:")
                print(results.failures)
            finally:
                await page.close()
                await browser.close()


if __name__ == "__main__":
    asyncio.run(load_from_wiki())
