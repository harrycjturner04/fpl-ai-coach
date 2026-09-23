# FPL AI Coach

An AI-assisted Fantasy Premier League coach. It combines a statistical/ML prediction layer, an integer linear programming (ILP) squad optimiser, and an LLM reasoning layer to recommend transfers, starting XI and captaincy — with explanations.

See [CLAUDE.md](CLAUDE.md) for the full architecture and roadmap.

## Status

- [x] Stage 1 — Ingestion skeleton (FPL API → raw JSON snapshots + Parquet tables)
- [ ] Stage 2 — ILP optimiser on current stats
- [ ] Stage 3 — Baseline prediction
- [ ] Stage 4 — Screenshot parsing
- [ ] Stage 5 — ML prediction model
- [ ] Stage 6 — LLM reasoning layer
- [ ] Stage 7 — Presentation

## Quick start

```bash
pip install -e ".[dev]"

python -m ingestion.cli                  # players, teams, gameweeks, fixtures
python -m ingestion.cli --history        # + per-player gameweek history (~670 requests, a few minutes)
python -m ingestion.cli --entry 1234567  # + your team: history, chips, transfers, squad prices

pytest
python scripts/validate_free_transfers.py   # re-check the free-transfer model against real managers
```

Your FPL team (entry) ID is the number in the URL of your team's "Points" page.

## Manager state (`--entry`)

The public FPL API doesn't expose free transfers, remaining chips or selling prices directly, so they are reconstructed in `ingestion/manager_state.py`:

| Output | How |
|---|---|
| Free transfers for next GW | Rollover model (+1/GW, cap 5; Wildcard/Free Hit keep the bank without adding one), re-anchored whenever a hit reveals the exact count. Validated against 2,885 real manager-gameweeks with zero mismatches. |
| Chips | Each season-half chip instance matched to when it was played → used / available / upcoming / expired. |
| Selling prices | Purchase price from the transfer log (Free Hit moves ignored), sell price = purchase + half the rise, rounded down. |
| Squad | The squad carried forward — i.e. the pre-Free-Hit squad if the last GW was a Free Hit. |

Written to `data/processed/manager_state_<id>.json` (compact summary for later stages) plus `entry_*` Parquet tables: overview, leagues, per-GW history, chips, transfers, past seasons, picks, squad.

Limitation: transfers already made for the upcoming GW aren't public until its deadline.

## Data layout

```
data/raw/<YYYY-MM-DD_HHMMSS>/*.json   immutable raw API snapshots (reproducible re-runs)
data/processed/*.parquet              tidy tables: players, teams, positions, gameweeks,
                                      fixtures, chips, player_history, entry_*
data/processed/game_rules.json        budget, squad composition, club cap, FT cap, sell-on fee
data/processed/manager_state_<id>.json
```

`data/` is gitignored — run the CLI to regenerate it.
