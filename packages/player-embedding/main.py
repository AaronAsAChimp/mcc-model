from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import numpy as np
import math
import json
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence

PLAYERS_PER_TEAM = 4
FEATURES_PER_PLAYER = 17
COINS_SCALE = 35000
FIRST_MCC_DATE = date.fromisoformat("2019-11-17")

#   Feature Description
# 1   win_rate    Fraction of events where their team placed 1st
# 2   dodgebolt_rate  Fraction where their team made top 2
# 3   avg_placement   1 - (mean coins-1)/9 (higher = better)
# 4   best_coins Best coins → inverted [0,1]
# 5   worst_coins Worst placement
# 6   placement_volatility    Std dev of coins (consistency)
# 7   recent_trend    Coins slope over last 3 events
# 8   n_participations    Log-scaled event count
# 9   time_since_first   log-scaled days since first mcc
# 10   recency Events in the past year
# 11  avg_teammate_overlap    Mean Jaccard similarity with prior teammates
# 12  is_returning_competitor 1 has played before, 0 is first time


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

    def __init__(self):
        self.event_count = 0
        self.placements = []
        self.team_coins = []
        self.player_coins = []
        self.events = []
        self.sum_team_similarity = 0
        self.previous_team = None
        self.current_team = None

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
                            player_tracker = PlayerTracker()
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
        self, players: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        team_size = len(players)
        team_tensor = torch.empty([1, team_size, FEATURES_PER_PLAYER])
        player_ids_tensor = torch.empty([team_size], dtype=torch.int32)

        for i in range(team_size):
            # the default of 0 is technically a existing player, but the the
            # is_returning_competitor should zero out everything.
            player_id = self._player_id_lookup.get(players[i], 0)
            player_tensor = self._players[player_id]
            team_tensor[0][i] = player_tensor
            player_ids_tensor[i] = player_id

        return (team_tensor, player_ids_tensor, torch.tensor([team_size]))


class TeamDataset(Dataset):
    _player_count: int
    _player_id_lookup: dict[str, int]
    _players: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    def __init__(self, extract_filename: str):
        self._players = []
        team_data: list[tuple[torch.Tensor, list[str], torch.Tensor]] = []

        tracker: dict[str, PlayerTracker] = {}

        with open(extract_filename, "r") as fd:
            for json_line in fd:
                event = json.loads(json_line)

                event_date = date.fromisoformat(event.get("date"))
                player_coin_lookup = event.get("player_coins")
                teams = event.get("teams")

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
                        player_name = players[i]
                        player_names.append(player_name)
                        player_tracker = tracker.get(player_name)
                        player_coins = player_coin_lookup.get(player_name)

                        if player_coins is None:
                            raise Exception(f"No coins for player {player_name}")

                        if player_tracker is None:
                            player_tracker = PlayerTracker()
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
                        (team_tensor, player_names, torch.tensor(team_coins))
                    )

        self._player_count = len(tracker)
        self._player_id_lookup = {}
        next_id = 0

        for player_name in tracker:
            self._player_id_lookup[player_name] = next_id
            next_id += 1

        for team_tensor, player_names, team_coins in team_data:
            player_ids = torch.tensor(
                [self._player_id_lookup[name] for name in player_names]
            )
            self._players.append((team_tensor, player_ids, team_coins))

    def __len__(self) -> int:
        return len(self._players)

    def __getitem__(self, key: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._players[key]

    def get_player_count(self) -> int:
        return self._player_count

    def get_config(self) -> dict[str, Any]:
        return {
            "player_count": self._player_count,
            "player_ids": self._player_id_lookup,
        }


class TeamPlacementPredictor(torch.nn.Module):
    def __init__(self, player_count: int, features_per_player=FEATURES_PER_PLAYER, embedding_dim=12):
        super().__init__()

        HIDDEN_LAYER_SIZE = round(FEATURES_PER_PLAYER * 2.5)
        OUTPUT_SIZE = 16

        self.player_embeddings = torch.nn.Embedding(player_count, embedding_dim)
        self.player_attention = torch.nn.Linear(OUTPUT_SIZE, 1)
        self.player_encoder = torch.nn.Sequential(
            torch.nn.LayerNorm(features_per_player),
            torch.nn.Linear(features_per_player, HIDDEN_LAYER_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_LAYER_SIZE, OUTPUT_SIZE),
            torch.nn.ReLU(),
        )
        combined_dim = embedding_dim + 16
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(combined_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(
        self,
        team_features: torch.Tensor,
        player_ids: torch.Tensor,
        team_sizes: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each player's feature vector: (batch, max_players, 16)
        encoded = self.player_encoder(team_features)

        # Mask out padded positions for both paths
        batch_size, max_players = team_features.shape[:2]
        device = team_features.device
        mask = torch.arange(max_players, device=device).unsqueeze(
            0
        ) < team_sizes.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()

        attn_logits = self.player_attention(encoded)
        attn_logits = attn_logits.squeeze(-1).masked_fill(~mask.squeeze(-1).bool(), float('-inf'))
        player_attention = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        team_encoded = (encoded * player_attention).sum(dim=1)

        embeds = self.player_embeddings(player_ids)
        team_embed = (embeds * mask).sum(dim=1) / team_sizes.unsqueeze(1).float()

        combined = torch.cat([team_embed, team_encoded], dim=1)
        return self.predictor(combined)


def collate_teams(batch):
    """Custom collate for variable-size teams (4 or 5 players)."""
    features = [item[0] for item in batch]
    player_ids = [item[1] for item in batch]
    coins = torch.stack([item[2] for item in batch])

    # Pad to the longest team in the batch
    features_padded = pad_sequence(features, batch_first=True)
    ids_padded = pad_sequence(player_ids, batch_first=True, padding_value=0)

    # Actual team sizes for correct mean pooling
    team_sizes = torch.tensor([f.size(0) for f in features])

    return features_padded, ids_padded, team_sizes, coins


EXTRACTS_FILENAME = "data/extracts.sorted.jsonl"
MODEL_FILENAME = "packages/player-embedding/model/model.pt"
CONFIG_FILENAME = "packages/player-embedding/model/config.json"


def train():
    num_epochs = 1000
    batch_size = 16
    report_modulus = 8
    print("Hello from player-embedding!")

    dataset = TeamDataset(EXTRACTS_FILENAME)

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

    for epoch in range(0, num_epochs):
        running_loss = 0
        last_loss = 0
        batch = 0

        model.train(True)

        for data in train_data:
            features, player_ids, team_sizes, labels = data

            batch += 1

            optimizer.zero_grad()

            outputs = model(features, player_ids, team_sizes)
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
                vfeatures, vplayer_ids, vteam_sizes, vlabels = vdata
                voutputs = model(vfeatures, vplayer_ids, vteam_sizes)

                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / total_test_batches
        print(f"LOSS train {last_loss} valid {avg_vloss}")

        torch.save(model.state_dict(), MODEL_FILENAME)

        with open(CONFIG_FILENAME, "w") as fd:
            json.dump(dataset.get_config(), fd)


def run():
    player_dataset = PlayerDataset(EXTRACTS_FILENAME, CONFIG_FILENAME, date.today())

    model = TeamPlacementPredictor(player_dataset.get_player_count())
    model.load_state_dict(torch.load(MODEL_FILENAME, weights_only=True))
    model.eval()

    event = {
        "teams": {
            # "placement": 1,
            "redrabbits": ["Feinberg", "Grian", "Shubble", "ZombieCleo"],
            # "placement": 5,
            "orangeocelots": ["acho", "Cambam010", "CherriFire", "Roscumber"],
            # "placement": 2,
            "yellowyaks": ["InTheLittleWood", "Kara Corvus", "Purpled", "Snifferish"],
            # "placement": 10,
            "limellamas": ["InfiniteDrift", "Mysticat", "Nukeri", "Shandrea"],
            # "placement": 9,
            "greengeckos": ["Graecie", "Legundo", "Mythical Sausage", "PrinceZam"],
            # "placement": 3,
            "cyancoyotes": ["CaptainSparklez", "ElainaExe", "GeminiTay", "4CVIT"],
            # "placement": 7,
            "aquaaxolotls": ["PeteZahHutt", "SadMilkman", "TapL", "vintage_applesauce"],
            # "placement": 6,
            "bluebats": ["Antfrost", "guqqie", "Rainhoe", "vGumiho"],
            # "placement": 8,
            "purplepandas": ["apokuna", "Couriway", "KatieGoBrr", "LexieMariex"],
            # "placement": 4,
            "pinkparrots": ["jojosolos", "Maezes", "millkberry", "Smajor1995"],
        }
    }

    results = []

    for [team_name, team] in event["teams"].items():
        [team_tensor, player_ids, team_size] = player_dataset.get_team_data(team)

        outputs = model(team_tensor, player_ids, team_size)

        results.append((team_name, outputs[0][0]))

    results = sorted(results, reverse=True, key=lambda result: result[1])

    for i in range(len(results)):
        [team, coins] = results[i]
        print(f"{i + 1}: { team } {coins * COINS_SCALE}")


def main():
    # train()
    run()


if __name__ == "__main__":
    main()
