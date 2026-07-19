Sorting the extracts:

```sh
jq -s -c "sort_by(.date)[]" data/extracts.jsonl > data/extracts.sorted.jsonl
```

```sh
uv run python packages/data-processing/main.py
```

```sh
uv run python packages/player-embedding/main.py 
```