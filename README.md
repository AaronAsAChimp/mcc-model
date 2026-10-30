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

Export the model and player features for the web app (rerun after retraining or
scraping new events):
```sh
uv run python packages/player-embedding/export_web.py
```

Run the web app:
```sh
cd web
npm run dev
```