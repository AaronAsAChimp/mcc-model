# Player Embedding

Learned vector representations for MCC players, trained end-to-end on a team placement prediction task. The trained embeddings can be plugged into downstream models (game outcome predictors, team balancers, etc.).

## Approach

### Architecture

A combined model that uses both hand-engineered features and learned per-player embeddings. The two components are trained jointly:

```
Input: 4–5 player names → [idx1, idx2, ...]

player_features = PlayerTracker(player).to_tensor(date)       # [13] per player
player_embeds   = Embedding(num_players, d)[idx]              # [d] per player

player_encoded  = MLP_encoder(player_features)                # [16] per player
team_encoded    = mean(player_encoded, dim=players)           # [16]
team_embed      = mean(player_embeds, dim=players)            # [d]

team_rep = concat(team_embed, team_encoded)                   # [12 + 16]

prediction = MLP(team_rep)                                    # scalar (placement)
```

Teams have 4 players normally; occasionally 5 (due to a late swap). Mean pooling over the player dimension makes the model invariant to team size — no padding or special-casing needed.

### Why learned embeddings on top of features?

The 13 features capture interpretable, aggregate stats, but they can't express soft patterns:

- "This player tends to overperform when teamed with specific other players"
- "This player has an intangible 'clutch factor' not captured by past placement"
- "This player's skill trajectory doesn't match their historical trend"

The learned embedding absorbs these residuals. Each known player gets a free vector that adjusts their prediction beyond what the features alone achieve.

### Cold-start players

New players are common in MCC. The architecture handles them without degradation:

- **Seen player**: `features + learned_embedding` — both contribute
- **Unseen player**: `features + zero_vector` — features carry the full prediction

The MLP learns to rely on features for the bulk of the prediction, so a zero embedding is harmless. The learned embedding only needs to represent the *residual* (deviation from feature-based prediction), which is an easier learning problem than learning from scratch.

### Alternative considered: feature → embedding encoder

An alternative is to replace the lookup table with a small MLP that projects features → embedding:

```
embedding = MLP_proj(features)   # works for any player
```

This guarantees every player gets a dense vector. The tradeoff is that known players no longer get free per-player parameters — they're constrained by the shared encoder, which limits expressiveness.

The current design (lookup + feature fallback) is preferred because:

- It produces a reusable artifact: a vector per known player that can be dropped into downstream models
- Known players get the full expressiveness of a free parameter
- Zero-vector fallback for unseen players is a common convention downstream models can handle

## Task

Predict each team's final placement (1–10) given the 4 players on the team. MSE loss on normalized placement. Alternatives considered:

- **Cross-entropy (top-2 vs not)**: simpler classification, but loses placement granularity
- **Ordinal regression**: better matches the ordered nature of placements, but more complex

MSE is a good starting point. Can be swapped once the pipeline is working.

## Artifact

The primary output is the `Embedding` weight matrix: a `[num_players × embedding_dim]` tensor.

Downstream consumers load this as a lookup table:

```python
player_id = player_vocab["PlayerName"]
embedding_vector = embedding_weights[player_id]  # [12]
```

For players not in the vocabulary, a zero vector is the convention. Downstream models should handle this (the zero-vector fallback is why features should remain available).

## Feature representation

`PlayerTracker` computes a 13-dim feature vector per player per event:

| # | Feature | Description |
|---|---|---|
| 1 | `win_rate` | Fraction of events where their team placed 1st |
| 2 | `dodgebolt_rate` | Fraction where their team made top 2 |
| 3 | `avg_placement` | 1 − (mean placement − 1) / 9 (inverted to [0,1]) |
| 4 | `best_placement` | Best placement, inverted |
| 5 | `worst_placement` | Worst placement, inverted |
| 6 | `placement_volatility` | Std dev of placement (consistency) |
| 7 | `recent_trend` | Placement slope over last 3 events |
| 8 | `n_participations` | Log-scaled event count |
| 9 | `time_since_first` | Log-scaled days since first MCC |
| 10 | `recency` | Events in the past year |
| 11 | `team_diversity` | Unique teams / total participations |
| 12 | `avg_teammate_overlap` | Mean Jaccard similarity with prior teammates |
| 13 | `is_returning_competitor` | 1 if played before, 0 if first time |

These features are available for *every* player — including first-timers — so the model never sees a cold-start player with zero information.

## Data

- Source: `data/extracts.sorted.jsonl` — 63 MCC events, ~398 unique players
- Each event yields 10 teams × 4 players = 630 team-level samples
    - Some teams hae 5 players because one was swapped out due to technical difficulties
- Chronological split (train on early events, validate/test on later events) — preserves time-series nature

## Training setup

- Optimizer: Adam (start LR at 1e-3)
- Embedding dimension: 12
- Model per-player encoder: `Linear(13, 32)` → ReLU → `Linear(32, 16)` → ReLU (shared across players)
- Model predictor: concat of mean-pooled `team_embed [12]` and mean-pooled `team_encoded [16]` → `Linear(28, 64)` → ReLU → `Linear(64, 1)`
- Mean pooling over the player dimension handles 4- and 5-player teams identically
- Early stopping on validation loss

## Remaining work

- [ ] Fix `best_placement` / `worst_placement` initialization bug (starts at 0, should start at first event value)
- [ ] Implement `team_diversity` feature (tensor[10])
- [ ] Chronological train/val/test split by event date
- [ ] Evaluation metrics (MAE, top-2 accuracy)
- [ ] Checkpoint saving (embedding weights as artifact)
- [ ] Type-check clean (pyright)
