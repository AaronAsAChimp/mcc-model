"""Export the trained TeamPlacementPredictor to ONNX for inference from JavaScript.

Run from the repo root:

    uv run python packages/player-embedding/export_onnx.py

The game axis is fixed at MAX_GAMES because the coin multiplier schedules
(COIN_MULTIPLIERS_6/8) are traced as length-8 constants; callers must pad
game id sequences to 8 with 0s. The player axis is dynamic since teams can
be 4 or 5 players.
"""

import json
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch

from games import GAME_IDS
from main import FEATURES_PER_PLAYER, MAX_GAMES, TeamPlacementPredictor

MODEL_PATH = "packages/player-embedding/model/model.pt"
CONFIG_PATH = "packages/player-embedding/model/config.json"
OUTPUT_PATH = "packages/player-embedding/model/model.onnx"

BATCH_SIZE = 4


def load_model() -> TeamPlacementPredictor:
    with open(CONFIG_PATH, "r") as fd:
        config = json.load(fd)

    model = TeamPlacementPredictor(config["player_count"])
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    return model


def make_inputs(batch_size: int = BATCH_SIZE) -> dict[str, torch.Tensor]:
    # Mixed team sizes so the export exercises the padding mask
    team_sizes = torch.tensor([4, 5, 4, 4], dtype=torch.long)[:batch_size]
    max_players = int(team_sizes.max())

    generator = torch.Generator().manual_seed(42)
    team_features = torch.randn(batch_size, max_players, FEATURES_PER_PLAYER, generator=generator)
    player_ids = torch.randint(0, 100, (batch_size, max_players), dtype=torch.long, generator=generator)

    # Game count varies per event; pad each row to MAX_GAMES with 0s (0 = padding).
    # Two rows get fewer games to cover the 6-game coin multiplier schedule.
    game_ids = torch.zeros(batch_size, MAX_GAMES, dtype=torch.long)
    for i in range(batch_size):
        count = 6 if i % 2 == 0 else MAX_GAMES
        ids = torch.randint(1, len(GAME_IDS) + 1, (count,), generator=generator)
        game_ids[i, :count] = ids

    return {
        "team_features": team_features,
        "player_ids": player_ids,
        "team_sizes": team_sizes,
        "game_ids": game_ids,
    }


def export(model: TeamPlacementPredictor, inputs: dict[str, torch.Tensor]) -> None:
    batch = torch.export.Dim("batch", min=1, max=256)
    players = torch.export.Dim("players", min=1, max=8)

    torch.onnx.export(
        model,
        tuple(inputs.values()),
        OUTPUT_PATH,
        input_names=list(inputs.keys()),
        output_names=["coins"],
        dynamic_shapes={
            "team_features": {0: batch, 1: players},
            "player_ids": {0: batch, 1: players},
            "team_sizes": {0: batch},
            "game_ids": {0: batch},
        },
        opset_version=18,
    )

    onnx.checker.check_model(OUTPUT_PATH)


def verify(model: TeamPlacementPredictor, inputs: dict[str, torch.Tensor]) -> None:
    # Re-run with different batch sizes and a different seed to prove the
    # exported graph is genuinely dynamic, not traced to BATCH_SIZE.
    extra = [inputs, make_inputs(batch_size=1), make_inputs(batch_size=2)]

    with torch.no_grad():
        expected = [model(*case.values()).squeeze(-1).numpy() for case in extra]

    session = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])

    for case, want in zip(extra, expected):
        got = session.run(["coins"], {name: tensor.numpy() for name, tensor in case.items()})[0].squeeze(-1)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-6)
        print(f"batch {len(want)}: max error {np.abs(got - want).max():.2e}")


def main() -> None:
    model = load_model()
    inputs = make_inputs()

    export(model, inputs)
    print(f"Exported {OUTPUT_PATH}")

    verify(model, inputs)
    print("Verification passed")


if __name__ == "__main__":
    sys.exit(main())
