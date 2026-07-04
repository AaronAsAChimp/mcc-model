Sorting the extracts:

```sh
jq -s -c "sort_by(.date)[]" data/extracts.jsonl > data/extracts.sorted.jsonl
```