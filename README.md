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

## License

The code in this repository is [MIT licensed](LICENSE).

The data in `data/` is **not** MIT licensed — it is derived from the
[MCC wiki](https://mcchampionship.fandom.com/wiki/MC_Championship_Wiki) and
remains [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). See
[data/README.md](data/README.md) for the attribution requirement.

Icons, fonts, and dependencies keep their original licenses; see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).