"""Export the web app's model assets.

Run from the repo root:

    uv run python packages/player-embedding/export_web.py

This writes:

- `web/public/model/model.onnx` - a single-file copy of the exported model,
  ready to fetch from the browser.
- `web/public/model/player-features.json` - each player's id and feature vector
  as of the export date. The browser needs these as model inputs; recompute
  them whenever the event data or model changes.
"""

import json
import os
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
WEB_MODEL_DIR = ROOT / "web" / "public" / "model"
PACKAGE_MODEL = ROOT / "packages" / "player-embedding" / "model" / "model.onnx"

# The training modules read their data files relative to the repo root.
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "packages" / "player-embedding"))

import onnx
import export_onnx
from main import CONFIG_FILENAME, EXTRACTS_FILENAME, FEATURES_PER_PLAYER, PlayerDataset
from shared_core.data import PlayerDataFile


def write_model() -> None:
    model = export_onnx.load_model()
    export_onnx.export(model, export_onnx.make_inputs())

    # `onnx.load` pulls external-data tensors into memory, so saving the result
    # produces one self-contained file the browser can fetch directly.
    graph = onnx.load(PACKAGE_MODEL)
    onnx.save_model(graph, WEB_MODEL_DIR / "model.onnx")

    print(f"Wrote {WEB_MODEL_DIR / 'model.onnx'}")


def write_player_features() -> None:
    with PlayerDataFile(str(ROOT / "data" / "players.json")) as player_data:
        dataset = PlayerDataset(EXTRACTS_FILENAME, CONFIG_FILENAME, date.today(), player_data)

    with open(CONFIG_FILENAME, "r") as fd:
        config = json.load(fd)

    # Features are indexed by the same ids the embedding table uses.
    rows = dataset._players.tolist()
    ids: dict[str, int] = config["player_ids"]

    players: dict[str, dict[str, object]] = {}

    for name in sorted(ids):
        values = [round(value, 6) for value in rows[ids[name]]]
        players[name] = {"id": ids[name], "features": values}

    # Unknown players cannot be selected in the web app, but keep a fallback so
    # the browser has the same behavior as the Python predictor.
    unknown = dataset._players.mean(dim=0).tolist()

    payload: dict[str, object] = {
        "generatedAt": date.today().isoformat(),
        "featureCount": FEATURES_PER_PLAYER,
        "unknown": [round(value, 6) for value in unknown],
        "players": players,
    }

    path = WEB_MODEL_DIR / "player-features.json"

    with open(path, "w") as fd:
        json.dump(payload, fd, separators=(",", ":"))

    print(f"Wrote {path} ({len(players)} players)")


def main() -> None:
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    write_model()
    write_player_features()


if __name__ == "__main__":
    main()
