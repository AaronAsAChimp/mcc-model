# Data

This data is **not** covered by the MIT license that applies to the code in this
repository. See [../LICENSE](../LICENSE).

## License

The event data here is derived from the
[MC Championship Wiki](https://mcchampionship.fandom.com/wiki/MC_Championship_Wiki),
which is licensed under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/),
and remains under that license.

If you redistribute this data, or a modified version of it, you must:

- credit the MC Championship Wiki and link to the license above
- distribute your version under CC BY-SA 3.0 (or a later compatible version)

Attribution string:

> Event data from the MC Championship Wiki (https://mcchampionship.fandom.com/wiki/MC_Championship_Wiki),
> licensed CC BY-SA 3.0.

## Files

- `extracts.jsonl` — scraped events, appended to incrementally by
  `packages/data-processing`.
- `extracts.sorted.jsonl` — the same records sorted by date, generated from
  `extracts.jsonl` (see the repo README for the command).
- `extracts.orig.jsonl` — an older snapshot kept for reference.
- `players.json`, `games.json`, `schemas/` — derived and hand-maintained
  reference data covering the same source.

Some player data also originates from the MCC Island API.
