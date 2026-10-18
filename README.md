Sorting the extracts:

```sh
jq -s -c "sort_by(.date)[]" data/extracts.jsonl > data/extracts.sorted.jsonl
```

Check the wiki for new data:

```sh
uv run python packages/data-processing/main.py
```

Run/train the model:
```sh
uv run python packages/player-embedding/main.py 
```