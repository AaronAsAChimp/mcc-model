import json
from typing import Mapping, NotRequired, Optional, TextIO, TypedDict


class ExtractsFile:
    _filename: str
    _fd: Optional[TextIO]

    def __init__(self, filename: str):
        self._filename = filename
        self._fd = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._fd is None:
            raise Exception("File not open")

        json_line = self._fd.readline()

        return json.loads(json_line)

    def __enter__(self):
        self._fd = open(self._filename, "r")

        return self

    def __exit__(self, type, value, traceback):
        if self._fd:
            self._fd.close()

        return True

class MccIslandGame(TypedDict):
    games: NotRequired[int | None]
    wins: NotRequired[int | None]

def mcci_win_percentage(stat: MccIslandGame):
    games = float(stat.get('games', 0))
    wins = float(stat.get('wins', 0))

    return wins / games if games != 0 else 0

class MccIslandStats(TypedDict):
    lastupdate: str
    battlebox: NotRequired[MccIslandGame | None]
    tgttos: NotRequired[MccIslandGame | None]
    skybattle: NotRequired[MccIslandGame | None]


class PandoraPlayerData(TypedDict):
    """
    The data from the Pandora's Box Competitions

    https://pandoras-box-event.fandom.com/wiki/
    """
    total_events: int
    events_won: int
    team_ranking: list[float]
    player_ranking: list[float]


class PlayerData(TypedDict):
    ign: NotRequired[Optional[str]]
    uuid: NotRequired[Optional[str]]
    aliases: NotRequired[list[str]]
    birthday: NotRequired[Optional[str]]
    island: NotRequired[Optional[MccIslandStats]]
    pandora: NotRequired[Optional[PandoraPlayerData]]

PlayerMapping = Mapping[str, PlayerData]

class PlayerDataFile:
    _filename: str | None
    _fd: Optional[TextIO]
    _data: PlayerMapping
    _name_lookup: dict[str, str]

    def __init__(self, filename: str | None = None):
        self._filename = filename
        self._fd = None
        self._name_lookup = {}

    def _build_lookup(self):
        lookup = {}

        for name, data in self._data.items():

            if name == '$schema':
                continue

            ign = data.get('ign')

            if ign is not None:
                lookup[ign.casefold()] = name

            for alias in data.get('aliases', []):
                lookup[alias.casefold()] = name

        self._name_lookup = lookup

    def __enter__(self):
        if self._filename is None:
            raise Exception('To be used as a context manager, PlayerDataFile must be initialized with a file name.')

        self._fd = open(self._filename, "r")

        self._data = json.load(self._fd)
        self._build_lookup()

        return self

    def __exit__(self, type, value, traceback):
        if self._fd:
            self._fd.close()

        # return True

    def __getitem__(self, key: str) -> Optional[PlayerData]:
        return self._data[key]

    def normalize_name(self, name: str) -> str:
        return self._name_lookup.get(name.casefold(), name)

    @classmethod
    def from_player_mapping(cls, mapping: PlayerMapping) -> PlayerDataFile:
        player_data = cls()

        player_data._data = mapping
        player_data._build_lookup()

        return player_data
