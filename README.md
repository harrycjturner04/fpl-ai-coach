# FPL AI Coach

An AI-assisted Fantasy Premier League coach. It combines a statistical/ML prediction layer, an integer linear programming (ILP) squad optimiser, and an LLM reasoning layer to recommend transfers, starting XI and captaincy — with explanations.

## Architecture

Five layers, each with a single responsibility; only clean, structured output crosses a layer boundary.

1. **Ingestion** — official FPL API (players, fixtures, per-player history, a manager's history/transfers/picks), later squad-screenshot parsing and injury/news sources.
2. **Feature engineering** — rolling form, per-90 rates, fixture difficulty, minutes reliability, set-piece duty, team strength.
3. **Prediction** — expected points per player for the next 1–5 gameweeks: a form × fixture-difficulty baseline, then gradient-boosted trees validated walk-forward by gameweek. Minutes/start probability is modelled separately.
4. **Optimisation** — a PuLP integer linear program: budget, 2/5/5/3 squad, max 3 per club, valid formation, captaincy, and the −4 transfer hit.
5. **Reasoning** — an LLM that turns news into structured risk flags, reviews the optimiser's output, plans chip timing, and writes the recommendation, calling the optimiser as a tool to test scenarios.

## Status

- [x] Stage 1 — Ingestion skeleton (FPL API → raw JSON snapshots + Parquet tables)
- [x] Stage 2 — ILP optimiser on current stats (from-scratch squads and transfer plans)
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

python -m optimisation.cli                   # best squad from scratch (£100m)
python -m optimisation.cli --entry 1234567   # best transfers for your team (after ingesting it)

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

## Optimiser

A single integer linear program (PuLP + CBC) chooses squad, starting XI, captain and transfers together, maximising
XI points + captain bonus + 0.1 × bench points − 4 × hits, subject to budget, 2/5/5/3 composition, max 3 per club and a
valid formation (limits read from FPL's own data). In transfer mode, owned players are valued at their selling price and
the solver only takes a hit when it gains more than 4 points; transfers are capped at free transfers + 2 by default.

Correctness is enforced three ways: an independent legality checker (`optimisation/validate.py`) re-checks every rule
after each solve; hand-built cases with known answers (club cap, budget, formation, hits, selling prices); and 50
randomised cross-checks against brute-force enumeration on a scaled-down game, for both modes.

Why an ILP: expected squad points are a sum of player expected points, so the problem is linear and the ILP returns the
proven optimum in under a second. Greedy, knapsack DP, genetic algorithms and RL were considered and rejected (not
exact, don't scale to these constraints, or need a season simulator).

Stage 2 scores players with `0.7 × ep_next + 0.3 × form`, scaled by chance of playing (`prediction/current_stats.py`) —
a placeholder until Stage 3's prediction model.

## Data layout

```
data/raw/<YYYY-MM-DD_HHMMSS>/*.json   immutable raw API snapshots (reproducible re-runs)
data/processed/*.parquet              tidy tables: players, teams, positions, gameweeks,
                                      fixtures, chips, player_history, entry_*
data/processed/game_rules.json        budget, squad composition, club cap, FT cap, sell-on fee
data/processed/manager_state_<id>.json
```

`data/` is gitignored — run the CLI to regenerate it.
