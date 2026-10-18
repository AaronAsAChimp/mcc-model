from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import numpy as np
import math
import json
from shared_core.data import PlayerDataFile
import torch
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from games import GAME_IDS

PLAYERS_PER_TEAM = 4
FEATURES_PER_PLAYER = 19
COINS_SCALE = 35000
FIRST_MCC_DATE = date.fromisoformat("2019-11-17")


def fg_color_rgb(r: int, g: int, b:int) -> str:
    return f'\033[38;2;{r};{g};{b}m'

def bg_color_rgb(r: int, g: int, b:int):
    return f'\033[48;2;{r};{g};{b}m'

def reset():
    return "\033[0m"

def console_color(text: str, fg: tuple[int, int, int] | None = None, bg: tuple[int, int, int] | None = None):
    return (fg_color_rgb(*fg) if fg is not None else '') + \
        (bg_color_rgb(*bg) if bg is not None else '') + \
        text + \
        (reset() if bg is not None or fg is not None else '')

#     Feature Description
# 1    win_rate    Fraction of events where their team placed 1st
# 2   dodgebolt_rate  Fraction where their team made top 2
# 3   avg_placement   1 - (mean coins-1)/9 (higher = better)
# 4   best_coins Best coins → inverted [0,1]
# 5   worst_coins Worst placement
# 6   placement_volatility    Std dev of coins (consistency)
# 7   recent_trend    Coins slope over last 3 events
# 8   n_participations    Log-scaled event count
# 9   time_since_first   log-scaled days since first mcc
# 10  recency Events in the past year
# 11  avg_teammate_overlap    Mean Jaccard similarity with prior teammates
# 12  is_returning_competitor 1 has played before, 0 is first time
# 13  age  log-scaled days
# 14  age_available  1 age is available, 0 if not


def jaccard_similarity(a: set, b: set) -> float:
    intersection = a.intersection(b)

    return len(intersection) / ((len(a) + len(b)) - len(intersection))


def scale_coins(val: float, total_events: int) -> float:
    return val / 35_000


@dataclass
class CoinStats:
    min: int
    max: int
    mean: float
    var: float

    @classmethod
    def from_list(cls, all_coins: list[int]):
        n = len(all_coins)

        if n == 0:
            return cls(mean=0, var=0, max=0, min=0)

        sum_team_coins = 0
        max = -float("inf")
        min = float("inf")

        for coins in all_coins:
            sum_team_coins += coins

            if coins < min:
                min = coins

            if coins > max:
                max = coins

        mean_team_coins = 0
        team_coin_variance = 0

        if n > 1:
            mean_team_coins = sum_team_coins / n
            team_coin_variance = sum((x - mean_team_coins) ** 2 for x in all_coins) / (
                n - 1
            )

        return cls(
            mean=mean_team_coins, var=team_coin_variance, max=int(max), min=int(min)
        )

with open('data/players.json', 'r') as player_json:
    PLAYER_DATA = json.load(player_json)

class PlayerTracker:
    event_count: int
    team_coins: list[int]
    placements: list[int]
    best_team_coins: int
    worst_team_coins: int
    events: list[date]
    sum_team_similarity: float
    previous_team: Optional[set[str]]
    current_team: Optional[set[str]]
    birthday: Optional[date]

    def __init__(self, name: str):
        self.event_count = 0
        self.placements = []
        self.team_coins = []
        self.player_coins = []
        self.events = []
        self.sum_team_similarity = 0
        self.previous_team = None
        self.current_team = None

        player_data = PLAYER_DATA.get(name, {})
        birthday = player_data.get('birthday')

        self.birthday = date.fromisoformat(birthday) if birthday is not None else None

    def update(
        self,
        new_team: set[str],
        placement: int,
        team_coins: int,
        player_coins: int,
        event_date: date,
    ):
        self.previous_team = self.current_team
        self.current_team = new_team

        if self.previous_team is not None:
            similarity = jaccard_similarity(
                self.previous_team,
                self.current_team,
            )

            self.sum_team_similarity += similarity

        self.event_count += 1
        self.placements.append(placement)
        self.team_coins.append(team_coins)
        self.player_coins.append(player_coins)

        self.events.append(event_date)

    def get_time_since_first(self, now: date) -> float:
        time_since_first = 0

        if len(self.events) > 0:
            log_player_days = math.log(max(1, (now - self.events[0]).days))
            log_total_days = math.log(max(1, (now - FIRST_MCC_DATE).days))
            time_since_first = log_player_days / log_total_days

        return time_since_first

    def to_tensor(self, now: date) -> torch.Tensor:
        tensor = torch.Tensor(FEATURES_PER_PLAYER)

        #
        # Placement
        #

        n = len(self.team_coins)
        wins = 0
        seconds = 0

        for placement in self.placements:
            if placement == 1:
                wins += 1
            elif placement == 2:
                seconds += 1

        #
        # Coins
        #

        team_coin_stats = CoinStats.from_list(self.team_coins)
        player_coin_stats = CoinStats.from_list(self.player_coins)

        #
        # Events
        #

        event_count = len(self.events)
        time_since_first = self.get_time_since_first(now)
        events_in_past_year = 0

        for event in self.events:
            time = now - event

            if time.days < 365:
                events_in_past_year += 1

        recency = events_in_past_year / 12  # MCC is roughly monthly

        team_coins_slope = 0
        player_coins_slope = 0

        if event_count > 3:
            event_2_back = self.events[-3]
            event_0_back = self.events[-1]

            time_between_events = (event_0_back - event_2_back).days

            team_coins_2_back = self.team_coins[-3]
            team_coins_0_back = self.team_coins[-1]

            team_coins_slope = (
                (team_coins_0_back - team_coins_2_back) / time_between_events
                if time_between_events != 0
                else 0
            )

            player_coins_2_back = self.player_coins[-3]
            player_coins_0_back = self.player_coins[-1]

            player_coins_slope = (
                (player_coins_0_back - player_coins_2_back) / time_between_events
                if time_between_events != 0
                else 0
            )

        #
        # Age
        # 

        age = None

        if self.birthday is not None:
            age = math.log((now - self.birthday).days)

        tensor[0] = (wins / n) if n != 0 else 0
        tensor[1] = ((wins + seconds) / n) if n != 0 else 0
        tensor[2] = scale_coins(team_coin_stats.mean, event_count)
        tensor[3] = scale_coins(team_coin_stats.max, event_count)
        tensor[4] = scale_coins(team_coin_stats.min, event_count)
        tensor[5] = math.sqrt(team_coin_stats.var)
        tensor[6] = team_coins_slope

        tensor[7] = scale_coins(player_coin_stats.mean, event_count)
        tensor[8] = scale_coins(player_coin_stats.max, event_count)
        tensor[9] = scale_coins(player_coin_stats.min, event_count)
        tensor[10] = math.sqrt(player_coin_stats.var)
        tensor[11] = player_coins_slope

        tensor[12] = math.log(n) if n > 0 else 0
        tensor[13] = time_since_first
        tensor[14] = recency
        tensor[15] = self.sum_team_similarity / event_count if event_count > 0 else 0
        tensor[16] = 1 if event_count > 0 else 0

        tensor[17] = age if age is not None else 0
        tensor[18] = 1 if age is not None else 0

        return tensor


class PlayerDataset(Dataset):
    _player_count: int
    _player_id_lookup: dict[str, int]
    _players: list[torch.Tensor]

    def __init__(self, extract_filename: str, config_filename: str, now: date):
        with open(config_filename, "r") as fd:
            config = json.load(fd)

        self._player_count = config["player_count"]
        self._player_id_lookup = config["player_ids"]

        # Force IDs to integers because JSON doesn't do integers
        for key in self._player_id_lookup:
            self._player_id_lookup[key] = int(self._player_id_lookup[key])

        tracker: dict[str, PlayerTracker] = {}

        with open(extract_filename, "r") as fd:
            for json_line in fd:
                event = json.loads(json_line)

                event_date = date.fromisoformat(event.get("date"))
                teams = event.get("teams")
                player_coin_lookup = event.get("player_coins")

                for team_name, team in teams.items():
                    placement = team.get("placement")
                    team_coins = team.get("coins") / COINS_SCALE
                    players = team.get("players")
                    team_size = len(players)

                    if team_size != 4:
                        print("team_size", team_name, team_size)

                    for i in range(team_size):
                        player_name = players[i]
                        player_tracker = tracker.get(player_name)

                        # If the player doesn't exist in the player coins they
                        # most likely are a new player.
                        #
                        # ... or something went terribly wrong in the data
                        # processing step.
                        player_coins = player_coin_lookup.get(player_name, 0)

                        if player_tracker is None:
                            player_tracker = PlayerTracker(player_name)
                            tracker[player_name] = player_tracker

                        player_tracker.update(
                            set(players),
                            placement,
                            team_coins,
                            player_coins,
                            event_date,
                        )

        self._players = [
            torch.empty(FEATURES_PER_PLAYER) for _ in range(self._player_count)
        ]

        for [name, player] in tracker.items():
            id = self._player_id_lookup[name]
            self._players[id] = player.to_tensor(now)

    def __len__(self) -> int:
        return len(self._players)

    def __getitem__(self, key: int) -> torch.Tensor:
        return self._players[key]

    def get_player_count(self) -> int:
        return self._player_count

    def get_team_data(
        self, players: list[str], player_data: PlayerDataFile
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        team_size = len(players)
        team_tensor = torch.empty([1, team_size, FEATURES_PER_PLAYER])
        player_ids_tensor = torch.empty([team_size], dtype=torch.int32)

        for i in range(team_size):
            player_name = player_data.normalize_name(players[i])
            player_id = self._player_id_lookup.get(player_name)

            if player_id is None:
                print(f'Missing data for player {player_name}')

            # the default of 0 is technically a existing player, but the the
            # is_returning_competitor should zero out everything.
            player_id = player_id if player_id is not None else 0
            player_tensor = self._players[player_id]
            team_tensor[0][i] = player_tensor
            player_ids_tensor[i] = player_id

        return (team_tensor, player_ids_tensor, torch.tensor([team_size]))


MAX_GAMES = 8  # excluding dodgebolt


def games_to_tensor(games: list[str]) -> torch.Tensor:
    # Shift IDs by 1 so that padding value 0 doesn't collide with acerace (raw ID 0)
    game_ids = [GAME_IDS[game] for game in games if game != 'dodgebolt']
    padded = game_ids + [0] * (MAX_GAMES - len(game_ids))

    return torch.tensor(padded, dtype=torch.long)


class TeamDataset(Dataset):
    _player_count: int
    _player_id_lookup: dict[str, int]
    _players: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(self, extract_filename: str, player_data: PlayerDataFile):
        self._players = []
        team_data: list[tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]] = []

        tracker: dict[str, PlayerTracker] = {}

        with open(extract_filename, "r") as fd:
            for json_line in fd:
                event = json.loads(json_line)

                event_date = date.fromisoformat(event.get("date"))
                player_coin_lookup = event.get("player_coins")
                teams = event.get("teams")
                games = event.get("games")

                # print('url:', event.get("wiki_url"))
                game_ids = games_to_tensor(games)

                for team_name, team in teams.items():
                    placement = team.get("placement")
                    team_coins = team.get("coins") / COINS_SCALE
                    players = team.get("players")
                    team_size = len(players)

                    if team_size != 4:
                        print("team_size", team_name, team_size)

                    team_tensor = torch.zeros(team_size, FEATURES_PER_PLAYER)
                    player_names: list[str] = []

                    for i in range(team_size):
                        player_name = player_data.normalize_name(players[i])
                        player_names.append(player_name)
                        player_tracker = tracker.get(player_name)
                        player_coins = player_coin_lookup.get(player_name)

                        if player_coins is None:
                            raise Exception(f"No coins for player {player_name}")

                        if player_tracker is None:
                            player_tracker = PlayerTracker(player_name)
                            tracker[player_name] = player_tracker

                        team_tensor[i] = player_tracker.to_tensor(event_date)

                        player_tracker.update(
                            set(players),
                            placement,
                            team_coins,
                            player_coins,
                            event_date,
                        )

                    team_data.append(
                        (team_tensor, player_names, game_ids, torch.tensor(team_coins))
                    )

        self._player_count = len(tracker)
        self._player_id_lookup = {}
        next_id = 0

        for player_name in tracker:
            self._player_id_lookup[player_name] = next_id
            next_id += 1

        for team_tensor, player_names, game_ids, team_coins in team_data:
            player_ids = torch.tensor(
                [self._player_id_lookup[name] for name in player_names]
            )
            self._players.append((team_tensor, player_ids, game_ids, team_coins))

    def __len__(self) -> int:
        return len(self._players)

    def __getitem__(self, key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._players[key]

    def get_player_count(self) -> int:
        return self._player_count

    def get_config(self) -> dict[str, Any]:
        return {
            "player_count": self._player_count,
            "player_ids": self._player_id_lookup,
        }


#
# Coin multiplier schedules by event format.
# Source: https://mcchampionship.fandom.com/wiki/Coin
#
COIN_MULTIPLIERS_6 = torch.tensor([1.0, 1.5, 2.0, 2.0, 2.5, 3.0, 0.0, 0.0])
COIN_MULTIPLIERS_8 = torch.tensor([1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 3.0])


class TeamPlacementPredictor(torch.nn.Module):
    def __init__(self, player_count: int):
        super().__init__()

        PLAYER_EMBEDDING_SIZE = 12
        PLAYER_HIDDEN_SIZE = round(FEATURES_PER_PLAYER * 2.5)
        PLAYER_ATTN_SIZE = 16

        TOTAL_GAME_COUNT = len(GAME_IDS)
        GAME_EMBEDDING_SIZE = 32

        PREDICTOR_SIZE = 64

        self.player_embeddings = torch.nn.Embedding(player_count, PLAYER_EMBEDDING_SIZE)
        self.player_attention = torch.nn.Linear(PLAYER_ATTN_SIZE, 1)
        self.player_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(FEATURES_PER_PLAYER),
            torch.nn.Linear(FEATURES_PER_PLAYER, PLAYER_HIDDEN_SIZE),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(PLAYER_HIDDEN_SIZE, PLAYER_ATTN_SIZE),
            torch.nn.ReLU(),
        )

        self.game_id_embedding = torch.nn.Embedding(TOTAL_GAME_COUNT + 1, GAME_EMBEDDING_SIZE, padding_idx=0)
        self.game_encoder = torch.nn.Sequential(
            torch.nn.Linear(GAME_EMBEDDING_SIZE, GAME_EMBEDDING_SIZE),
            torch.nn.ReLU()
        )

        combined_dim = PLAYER_EMBEDDING_SIZE + PLAYER_ATTN_SIZE + GAME_EMBEDDING_SIZE
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, PREDICTOR_SIZE),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(PREDICTOR_SIZE, 1),
        )

    def forward(
        self,
        team_features: torch.Tensor,
        player_ids: torch.Tensor,
        team_sizes: torch.Tensor,
        game_ids: torch.Tensor
    ) -> torch.Tensor:
        # Encode each player's feature vector: (batch, max_players, 16)
        encoded = self.player_encoder(team_features)

        # Mask out padded positions for both paths
        batch_size, max_players = team_features.shape[:2]
        device = team_features.device
        player_mask = torch.arange(max_players, device=device).unsqueeze(
            0
        ) < team_sizes.unsqueeze(1)
        player_mask = player_mask.unsqueeze(-1).float()

        attn_logits = self.player_attention(encoded)
        attn_logits = attn_logits.squeeze(-1).masked_fill(~player_mask.squeeze(-1).bool(), float('-inf'))
        player_attention = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        team_encoded = (encoded * player_attention).sum(dim=1)

        individual_player_embeds = self.player_embeddings(player_ids)
        team_embed = (individual_player_embeds * player_mask).sum(dim=1) / team_sizes.unsqueeze(1).float()

        game_embeds = self.game_id_embedding(game_ids)
        game_mask = (game_ids != 0)  # (batch, max_games)

        # Select the correct coin multiplier schedule per event, then
        # zero out padding positions so they don't contribute to the sum.
        num_games = game_mask.sum(dim=1)  # (batch,)
        is_6_game = (num_games == 6).unsqueeze(1)  # (batch, 1)
        weights = torch.where(
            is_6_game,
            COIN_MULTIPLIERS_6.to(device),
            COIN_MULTIPLIERS_8.to(device),
        )
        weights = (weights * game_mask.float()).unsqueeze(-1)  # (batch, max_games, 1)

        game_embed = (game_embeds * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        game_embed = self.game_encoder(game_embed)

        combined = torch.cat([team_embed, team_encoded, game_embed], dim=1)
        return self.predictor(combined)


def collate_teams(batch):
    """Custom collate for variable-size teams (4 or 5 players)."""
    features = [item[0] for item in batch]
    player_ids = [item[1] for item in batch]
    game_ids = [item[2] for item in batch]
    coins = torch.stack([item[3] for item in batch])

    # Pad to the longest team in the batch
    features_padded = pad_sequence(features, batch_first=True)
    players_padded = pad_sequence(player_ids, batch_first=True, padding_value=0)

    # Pad to the greatest number games in an event
    games_padded = pad_sequence(game_ids, batch_first=True, padding_value=0)

    # Actual team sizes for correct mean pooling
    team_sizes = torch.tensor([f.size(0) for f in features])

    return features_padded, players_padded, team_sizes, games_padded, coins


EXTRACTS_FILENAME = "data/extracts.sorted.jsonl"
MODEL_FILENAME = "packages/player-embedding/model/model.pt"
CONFIG_FILENAME = "packages/player-embedding/model/config.json"
TRAINING_LOG_FILENAME = "train-log.csv"

def train():
    num_epochs = 1000
    batch_size = 16
    report_modulus = 8
    print("Hello from player-embedding!")
    dataset = None

    with PlayerDataFile('data/players.json') as player_data:
        dataset = TeamDataset(EXTRACTS_FILENAME, player_data)

    player_count = dataset.get_player_count()
    print(f"Players: {player_count}")

    dataset_size = len(dataset)
    train_size = math.floor(dataset_size * 0.8)
    test_size = dataset_size - train_size

    print(f"Total size: {dataset_size}, Train size: {train_size}, Test size: {test_size}")

    # [train_set, test_set] = random_split(dataset, [0.80, 0.20])
    train_set = Subset(dataset, range(0, train_size))
    test_set = Subset(dataset, range(train_size, train_size + test_size))
    total_test_batches = math.ceil(len(test_set) / batch_size)

    train_data = DataLoader(
        train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_teams
    )
    test_data = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_teams)
    model = TeamPlacementPredictor(player_count)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    with open(TRAINING_LOG_FILENAME, 'w') as log_fd:
        log_fd.write('batch,training loss,validation loss\n')
        best_vloss = float('inf')

        for epoch in range(0, num_epochs):
            running_loss = 0
            last_loss = 0
            batch = 0

            model.train(True)

            for data in train_data:
                features, player_ids, team_sizes, games_multi_hot, labels = data

                batch += 1

                optimizer.zero_grad()

                outputs = model(features, player_ids, team_sizes, games_multi_hot).squeeze(-1)
                loss = loss_fn(outputs, labels)

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if batch % report_modulus == 0:
                    last_loss = running_loss / report_modulus
                    print(f"  batch {epoch + 1} loss: {last_loss}")
                    running_loss = 0

            running_vloss = 0.0

            model.eval()

            with torch.no_grad():
                for vdata in test_data:
                    vfeatures, vplayer_ids, vteam_sizes, vgame_multi_hot, vlabels = vdata
                    voutputs = model(vfeatures, vplayer_ids, vteam_sizes, vgame_multi_hot).squeeze(-1)

                    vloss = loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / total_test_batches
            improved = ''

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(model.state_dict(), MODEL_FILENAME)
                improved = console_color(' ✓', fg=(0, 255, 0))

            print(f"LOSS train {last_loss} valid {avg_vloss}{improved}")
            log_fd.write(f'{epoch + 1},{last_loss},{avg_vloss}\n')

            with open(CONFIG_FILENAME, "w") as fd:
                json.dump(dataset.get_config(), fd)


def run():
    player_dataset = PlayerDataset(EXTRACTS_FILENAME, CONFIG_FILENAME, date.today())

    model = TeamPlacementPredictor(player_dataset.get_player_count())
    model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    model.eval()

    event = {
        "games": [
            "buildmart",
            "battlebox",
            "parkourwarriorsurvivor",
            "bingobutfast",
            "acerace",
            "skybattle",
            "holeinthewall",
            "survivalgames",
        ],

        "teams": {
            # "placement": 1,
            "redrabbits": ["aimsey", "bekyamon", "ghostiefruit", "hahhahxxrose"],
            # "placement": 5,
            "orangeocelots": ["cubfan135", "ish13c", "Ph1Lza", "Seapeekay"],
            # "placement": 2,
            "yellowyaks": ["FalseSymmetry", "InTheLittleWood", "Kratzy", "PeteZahHutt"],
            # "placement": 10,
            "limellamas": ["4CVIT", "Smajor1995", "Sneegsnag", "xChocoBars"],
            # "placement": 9,
            "greengeckos": ["Feinberg", "GoodTimesWithScar", "impulseSV", "Tubbo"],
            # "placement": 3,
            "cyancoyotes": ["AntVenom", "ashswag", "SB737", "Shadoune666"],
            # "placement": 7,
            "aquaaxolotls": ["Antfrost", "sleepy", "Shubble", "Steven Suptic"],
            # "placement": 6,
            "bluebats": ["CaptainSparkles", "ElainaExe", "fruitberries", "Kara Corvus"],
            # "placement": 8,
            "purplepandas": ["Michaelmcchill", "PrinceZam", "Purpled", "Snifferish"],
            # "placement": 4,
            "pinkparrots": ["Cambam", "FireBreathMan", "Grian", "Rendog"],
        }
    }

    results = []
    game_ids = games_to_tensor(event.get("games", [])).unsqueeze(0)

    with PlayerDataFile('data/players.json') as player_data:
        for [team_name, team] in event["teams"].items():
            [team_tensor, player_ids, team_size] = player_dataset.get_team_data(team, player_data)

            outputs = model(team_tensor, player_ids, team_size, game_ids).squeeze(-1)

            results.append((team_name, outputs[0]))

    results = sorted(results, reverse=True, key=lambda result: result[1])

    for i in range(len(results)):
        [team, coins] = results[i]
        print(f"{i + 1}: { team } {coins * COINS_SCALE}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparser = parser.add_subparsers(required=True)

    train_parser = subparser.add_parser('train')
    train_parser.set_defaults(func=train)

    run_parser = subparser.add_parser('run')
    run_parser.set_defaults(func=run)

    args = parser.parse_args()

    print(args.func())
