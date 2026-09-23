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
python -m ingestion.cli --entry 1234567  # + your team's current picks (your FPL team ID)

pytest
```

## Data layout

```
data/raw/<YYYY-MM-DD_HHMMSS>/*.json   immutable raw API snapshots (reproducible re-runs)
data/processed/*.parquet              tidy tables: players, teams, positions, gameweeks,
                                      fixtures, player_history, entry_picks
```

`data/` is gitignored — run the CLI to regenerate it.
