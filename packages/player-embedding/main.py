from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import numpy as np
import math
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence

PLAYERS_PER_TEAM = 4
FEATURES_PER_PLAYER = 13
FIRST_MCC_DATE = date.fromisoformat("2019-11-17")

#   Feature Description
# 1   win_rate    Fraction of events where their team placed 1st
# 2   dodgebolt_rate  Fraction where their team made top 2
# 3   avg_placement   1 - (mean placement-1)/9 (higher = better)
# 4   best_placement  Best placement → inverted [0,1]
# 5   worst_placement Worst placement
# 6   placement_volatility    Std dev of placement (consistency)
# 7   recent_trend    Placement slope over last 3 events
# 8   n_participations    Log-scaled event count
# 9   time_since_first   log-scaled days since first mcc
# 10   recency Events in the past year
# 11  team_diversity  Unique teams / total participations
# 12  avg_teammate_overlap    Mean Jaccard similarity with prior teammates
# 13  is_returning_competitor 1 has played before, 0 is first time


def jaccard_similarity(a: set, b: set) -> float:
    intersection = a.intersection(b)

    return len(intersection) / ((len(a) + len(b)) - len(intersection))


def scale_placement(val: float, total_events: int) -> float:
    return (1 - (val - 1) / 9) if total_events != 0 else 0


class PlayerTracker:
    event_count: int
    placements: list[int]
    best_placement: int
    events: list[date]
    sum_team_similarity: float
    previous_team: Optional[set[str]]
    current_team: Optional[set[str]]

    def __init__(self):
        self.event_count = 0
        self.placements = []
        self.best_placement = 0
        self.worst_placement = 0
        self.events = []
        self.sum_team_similarity = 0
        self.previous_team = None
        self.current_team = None

    def update(self, new_team: set[str], placement: int, event_date: date):
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

        if placement < self.best_placement:
            self.best_placement = placement

        if placement > self.worst_placement:
            self.worst_placement = placement

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

        wins = 0
        seconds = 0
        sum_placements = 0

        for placement in self.placements:
            if placement == 1:
                wins += 1
            elif placement == 2:
                seconds += 1

            sum_placements += placement

        n = len(self.placements)

        mean_placement = 0
        variance = 0

        if n > 1:
            mean_placement = sum_placements / n
            variance = sum((x - mean_placement) ** 2 for x in self.placements) / (n - 1)

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

        placement_slope = 0

        if event_count > 3:
            event_2_back = self.events[-3]
            event_0_back = self.events[-1]

            placement_2_back = self.placements[-3]
            placement_0_back = self.placements[-1]

            time_between_events = (event_0_back - event_2_back).days

            placement_slope = (
                (placement_0_back - placement_2_back) / time_between_events
                if time_between_events != 0
                else 0
            )

        tensor[0] = (wins / n) if n != 0 else 0
        tensor[1] = ((wins + seconds) / n) if n != 0 else 0
        tensor[2] = scale_placement(mean_placement, event_count)
        tensor[3] = scale_placement(self.best_placement, event_count)
        tensor[4] = scale_placement(self.worst_placement, event_count)
        tensor[5] = math.sqrt(variance)
        tensor[6] = placement_slope
        tensor[7] = math.log(n) if n > 0 else 0
        tensor[8] = time_since_first
        tensor[9] = recency
        tensor[10] = 0  # TODO
        tensor[11] = self.sum_team_similarity / event_count if event_count > 0 else 0
        tensor[12] = 1 if event_count > 0 else 0

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

                for team_name, team in teams.items():
                    placement = team.get("placement")
                    players = team.get("players")
                    team_size = len(players)

                    if team_size != 4:
                        print("team_size", team_name, team_size)

                    for i in range(team_size):
                        player_name = players[i]
                        player_tracker = tracker.get(player_name)

                        if player_tracker is None:
                            player_tracker = PlayerTracker()
                            tracker[player_name] = player_tracker

                        player_tracker.update(set(players), placement, event_date)

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
                teams = event.get("teams")

                for team_name, team in teams.items():
                    placement = team.get("placement")
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

                        if player_tracker is None:
                            player_tracker = PlayerTracker()
                            tracker[player_name] = player_tracker

                        team_tensor[i] = player_tracker.to_tensor(event_date)

                        player_tracker.update(set(players), placement, event_date)

                    team_data.append(
                        (team_tensor, player_names, torch.tensor(placement))
                    )

        self._player_count = len(tracker)
        self._player_id_lookup = {}
        next_id = 0

        for player_name in tracker:
            self._player_id_lookup[player_name] = next_id
            next_id += 1

        for team_tensor, player_names, placement in team_data:
            player_ids = torch.tensor(
                [self._player_id_lookup[name] for name in player_names]
            )
            self._players.append((team_tensor, player_ids, placement))

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
    def __init__(self, player_count: int, features_per_player=13, embedding_dim=12):
        super().__init__()
        self.player_embeddings = torch.nn.Embedding(player_count, embedding_dim)
        self.player_encoder = torch.nn.Sequential(
            torch.nn.Linear(features_per_player, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
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

        team_encoded = (encoded * mask).sum(dim=1) / team_sizes.unsqueeze(1).float()

        embeds = self.player_embeddings(player_ids)
        team_embed = (embeds * mask).sum(dim=1) / team_sizes.unsqueeze(1).float()

        combined = torch.cat([team_embed, team_encoded], dim=1)
        return self.predictor(combined)


def collate_teams(batch):
    """Custom collate for variable-size teams (4 or 5 players)."""
    features = [item[0] for item in batch]
    player_ids = [item[1] for item in batch]
    placements = torch.stack([item[2] for item in batch])

    # Pad to the longest team in the batch
    features_padded = pad_sequence(features, batch_first=True)
    ids_padded = pad_sequence(player_ids, batch_first=True, padding_value=0)

    # Actual team sizes for correct mean pooling
    team_sizes = torch.tensor([f.size(0) for f in features])

    return features_padded, ids_padded, team_sizes, placements


EXTRACTS_FILENAME = "../../data/extracts.sorted.jsonl"
MODEL_FILENAME = "./model/model.pt"
CONFIG_FILENAME = "./model/config.json"


def train():
    num_epochs = 1000
    batch_size = 16
    report_modulus = 8
    print("Hello from player-embedding!")

    dataset = TeamDataset("../../data/extracts.sorted.jsonl")

    player_count = dataset.get_player_count()
    print(f"Players: {player_count}")

    [train_set, test_set] = random_split(dataset, [0.80, 0.20])
    total_test_batches = math.ceil(len(test_set) / batch_size)

    train_data = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_teams
    )
    test_data = DataLoader(test_set, batch_size=16, collate_fn=collate_teams)
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

    results = sorted(results, key=lambda result: result[1])

    for i in range(len(results)):
        print(f"{i + 1}: {results[i][0]} {results[i][1]}")


def main():
    # train()
    run()


if __name__ == "__main__":
    main()
