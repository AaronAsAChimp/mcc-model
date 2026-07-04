# MCC Model

**MC Championship (MCC)** is a monthly invite-only Minecraft tournament organized by Noxcrew and Scott Major. 10 teams of 4 content creators compete in 8 minigames (parkour, combat, teamwork, etc.) for coins; the top 2 teams advance to a final Dodgebolt round. This repo scrapes MCC data and will train a model to predict event outcomes.

## Package manager

- **UV** — workspace root at `pyproject.toml`, members in `packages/*`.
- Every command runs from repo root. No `cd` needed.

## Python

- **Python 3.14+** required (`.python-version`, `requires-python = ">=3.14"`).
- Virtual env: `.venv/` in repo root. Activate with `source .venv/bin/activate` or use `uv run`.

## Type checking

`pyrightconfig.json` exists at root — run `uv run pyright` from repo root.

No linter, formatter, or test framework is configured.

## Data

- `data/extracts.jsonl` — scraped MCC event data (63 events), appended to incrementally.
- `data/extracts.sorted.jsonl` — same data sorted by date. Regenerate with:
  ```sh
  uv run jq -s -c "sort_by(.date)[]" data/extracts.jsonl > data/extracts.sorted.jsonl
  ```

## Packages

### `packages/data-processing`

Web scraper for `mcchampionship.fandom.com` using Playwright. Run from repo root:

```sh
uv run python packages/data-processing/main.py
```

Quirks:
- Scrapes sequentially with `time.sleep(10)` between pages (rate limiting).
- Hardcoded to process max 10 new URLs per run (line 449 slice `[0:10]`).
- Skips URLs already present in `data/extracts.jsonl` (dedup on `wiki_url`).
- Output appends to `data/extracts.jsonl`.

**Install Playwright browsers before first run:**

```sh
uv run playwright install firefox
```

### `packages/player-embedding`

Stub — `main.py` just prints "Hello from player-embedding!". Not yet implemented.

## Data model

Event records are serialized as JSONL. Each line is an `MccEvent`:
- `name`, `wiki_url`, `date` (ISO 8601)
- `teams` — map of team slugs to player name arrays
- `winning_team`, `second_place_team` — team slugs
- `games` — array of game slugs

Team and game enums are defined in `packages/data-processing/main.py` with string-to-slug mapping — check those before adding new values.
