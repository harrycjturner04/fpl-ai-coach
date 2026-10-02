# FPL AI Coach

An AI-assisted Fantasy Premier League coach. It combines a statistical/ML prediction layer, an integer linear programming (ILP) squad optimiser, and an LLM reasoning layer to recommend transfers, starting XI and captaincy; with explanations.

## Architecture

Five layers, each with a single responsibility; only clean, structured output crosses a layer boundary.

1. **Ingestion**: official FPL API (players, fixtures, per-player history, a manager's history/transfers/picks), later squad-screenshot parsing and injury/news sources.
2. **Feature engineering**: rolling form, per-90 rates, fixture difficulty, minutes reliability, set-piece duty, team strength.
3. **Prediction**: expected points per player for the next 1–5 gameweeks, built component-by-component from team attack/defence ratings, player minutes/start probability and per-90 rates, combined through FPL's own scoring rules; next, gradient-boosted trees validated walk-forward by gameweek. Minutes/start probability is modelled separately.
4. **Optimisation**: a PuLP integer linear program covering budget, 2/5/5/3 squad, max 3 per club, valid formation, captaincy, and the −4 transfer hit.
5. **Reasoning**: an LLM that turns news into structured risk flags, reviews the optimiser's output, plans chip timing, and writes the recommendation, calling the optimiser as a tool to test scenarios.

## Status

- [x] Stage 1: Ingestion skeleton (FPL API → raw JSON snapshots + Parquet tables)
- [x] Stage 2: ILP optimiser on current stats (from-scratch squads and transfer plans)
- [x] Stage 3a: Component prediction model (appearance, goals, assists, clean sheets, bonus, etc.), live and feeding the optimiser by default
- [ ] Stage 3b: Multi-week optimiser horizon
- [ ] Stage 4: Screenshot parsing
- [ ] Stage 5: ML prediction model
- [ ] Stage 6: LLM reasoning layer
- [ ] Stage 7: Presentation

## Quick start

```bash
pip install -e ".[dev]"

python -m ingestion.cli                  # players, teams, gameweeks, fixtures
python -m ingestion.cli --history        # + per-player gameweek history (~670 requests, a few minutes)
python -m ingestion.cli --entry 1234567  # + your team: history, chips, transfers, squad prices
python -m ingestion.archive              # past-season archive for training/backtesting (once per season)

python -m prediction.cli                     # expected points for the next 5 gameweeks
python -m prediction.backtest                # walk-forward backtest of the component model vs. benchmarks
python -m prediction.tune                    # coordinate-descent tuning on the tuning seasons

# The optimiser re-pulls data automatically if it's over 24h old or a deadline has passed.
python -m optimisation.cli                   # best squad from scratch (£100m), scored by the component model
python -m optimisation.cli --entry 1234567   # best transfers for your team (after ingesting it)

pytest                                      # offline tests
pytest -m live                              # check the real FPL API still has every field we use
python scripts/validate_free_transfers.py   # re-check the free-transfer model against real managers
```

Your FPL team (entry) ID is the number in the URL of your team's "Points" page.

At the end of a season, add it to `ARCHIVE_SEASONS` in `ingestion/archive.py` and re-run `python -m ingestion.archive`: the current-season rows reset once the season label changes, so last season's data only survives in the archive.

## Manager state (`--entry`)

The public FPL API doesn't expose free transfers, remaining chips or selling prices directly, so they are reconstructed in `ingestion/manager_state.py`:

| Output | How |
|---|---|
| Free transfers for next GW | Rollover model (+1/GW, cap 5; Wildcard/Free Hit keep the bank without adding one), re-anchored whenever a hit reveals the exact count. Validated against 2,885 real manager-gameweeks with zero mismatches. |
| Chips | Each season-half chip instance matched to when it was played → used / available / upcoming / expired. |
| Selling prices | Purchase price from the transfer log (Free Hit moves ignored), sell price = purchase + half the rise, rounded down. |
| Squad | The squad carried forward, i.e. the pre-Free-Hit squad if the last GW was a Free Hit. |

Written to `data/processed/manager_state_<id>.json` (compact summary for later stages) plus `entry_*` Parquet tables: overview, leagues, per-GW history, chips, transfers, past seasons, picks, squad.

Limitation: transfers already made for the upcoming GW aren't public until its deadline.

## Optimiser

A single integer linear program (PuLP + CBC) chooses squad, starting XI, captain and transfers together, maximising
XI points + captain bonus + 0.1 × bench points − 4 × hits, subject to budget, 2/5/5/3 composition, max 3 per club and a
valid formation (limits read from FPL's own data). In transfer mode, owned players are valued at their selling price and
the solver only takes a hit when it gains more than 4 points; transfers are capped at free transfers + 2 by default.
Next week's free-transfer bank is part of the model: each free transfer rolled over (up to the cap of 5) is worth
`--ft-value` points (default 1.5), so the optimiser won't spend a transfer on a tiny gain. This is the single-week case
of the planned multi-week optimiser, where the same value prices free transfers left at the end of the horizon.

Correctness is enforced three ways: an independent legality checker (`optimisation/validate.py`) re-checks every rule
after each solve; hand-built cases with known answers (club cap, budget, formation, hits, selling prices); and 50
randomised cross-checks against brute-force enumeration on a scaled-down game, for both modes.

Why an ILP: expected squad points are a sum of player expected points, so the problem is linear and the ILP returns the
proven optimum in under a second. Greedy, knapsack DP, genetic algorithms and RL were considered and rejected (not
exact, don't scale to these constraints, or need a season simulator).

The optimiser now scores players with the Stage 3a component model of expected points by default (see
docs/stage-3a-prediction-design.md); pass `--scorer placeholder` to fall back to the Stage 2 ep_next/form blend.

## Prediction results

Walk-forward backtest on the held-out 2025/26 season (parameters tuned on 2022/23 to 2024/25 only; 2025/26 was
evaluated once). Lower RMSE (primary) and MAE are better; higher rank correlation (rho, within position) is better;
decision value is the average real points per gameweek of the XI and captain the optimiser picks from each model's
predictions.

| Model | RMSE (next GW) | MAE (next GW) | rho (next GW) | Decision value | RMSE (5 GWs ahead) |
|---|---|---|---|---|---|
| Component model | 2.660 | 1.752 | 0.521 | 57.8 | 2.803 |
| Rebuilt FPL form | 2.999 | 1.990 | 0.446 | 48.1 | 3.108 |
| Naive (last 5 average) | 2.908 | 1.986 | 0.410 | 47.5 | 2.989 |

At the next gameweek, the component model's RMSE is 0.345 lower than rebuilt FPL form (95% bootstrap interval -0.403
to -0.295) and its rank correlation is 0.075 higher (95% bootstrap interval 0.042 to 0.103); both intervals sit
entirely on the side that favours the component model, so the gain over form is unlikely to be noise. FPL's own `xP`
from the archive was excluded as a benchmark because the archive captures it after each gameweek is played rather
than before the deadline, so it already reflects the result it should be forecasting (design doc section 10.1).
Known limitation: the archive holds the final fixture schedule, so predictions more than one week ahead see double
gameweeks slightly earlier than they were announced. Defensive contribution scoring exists only from 2025/26, so
this holdout is also its first evaluation.

The RMSE in the table is the mean of each gameweek's RMSE; the quoted 0.345 difference above (and its confidence
interval) instead uses the pooled root-mean-square of the per-gameweek RMSEs, a different (quadratic, not
arithmetic) average that weights worse gameweeks more heavily, so the two numbers are not directly comparable.

Change after the holdout: these results used a 5-day memory for the minutes model, which tuning chose on
next-gameweek accuracy alone. A sensitivity check on the tuning seasons (never the holdout) showed that 5 days
hurts predictions two or more weeks ahead, because a player rested once is usually back soon after: a 10-day memory
matches 5 days on next-gameweek RMSE and beats it at every later horizon. The shipped model therefore uses 10 days.
The holdout was not re-run, to keep it a single, untouched evaluation; a horizon-dependent minutes model is planned
for Stage 3b.

Reproduce the holdout: `python -m prediction.backtest --seasons 2025-26 --models naive,form,component --params
data/processed/model_params.json` (`--models` defaults to `naive,form`; add `component` explicitly, with `--params`
pointing at a tuned parameters file, to include it).

## How this project is built

I use [Claude Code](https://claude.com/claude-code) as a development tool, in the same way I'd use any tool that makes me faster: it writes much of the implementation and tests, and runs an automated review pass. The thinking behind the project is mine. I set the vision and architecture, decide the approach for each problem, and choose between alternatives after weighing them up. The reasoning behind the key decisions, such as why the optimiser is an integer linear program and how free transfers are valued, is documented in the sections above.

The most important parts of the system, the theory and maths behind the decision making and the research into which approaches are best, are driven by meticulous direction from me rather than left to the tool. Every change is planned with me before any code is written; I review the code manually on top of the automated review, and I make my own edits to it directly. Correctness is backed by evidence rather than trust: the optimiser is cross-checked against brute-force enumeration, an independent validator re-checks every FPL rule after each solve, the free-transfer model was validated against thousands of real manager-gameweeks, and the reconstructed team data was checked against my own FPL account. Commits that Claude contributed to are marked with a `Co-Authored-By` line.

## Data layout

```
data/raw/<YYYY-MM-DD_HHMMSS>/*.json   raw API responses from the latest pull only (older pulls pruned)
data/processed/*.parquet              tidy tables: players, teams, positions, gameweeks,
                                      fixtures, chips, player_history, entry_*
data/processed/archive_match_log.parquet    past-season match-log rows (ingestion/archive.py)
data/processed/current_season_rows.parquet  this season's match-log rows, refreshed each pull
data/processed/match_log.parquet      archive + current-season rows combined (deduplicated by season):
                                      the prediction model's input
data/processed/ep_next_log.parquet    FPL's own ep_next forecast, snapshotted before each deadline
data/processed/predictions.parquet    the component model's latest live output (prediction/cli.py)
data/processed/model_params.json      tuned parameters from a local re-tune (prediction/tune.py); takes
                                      priority over the copy shipped at prediction/model_params.json
data/processed/tuning_log.parquet     one row per coordinate-descent step (prediction/tune.py)
data/processed/game_rules.json        budget, squad composition, club cap, FT cap, sell-on fee
data/processed/metadata.json          when data and each team were last pulled (drives auto-refresh)
data/processed/manager_state_<id>.json
prediction/model_params.json          tuned parameters shipped in the repo, so a fresh clone uses them
                                      without a local re-tune
```

`data/` is not committed; run the CLI to regenerate it. Current-season player prices (`players.price`, and
`current_season_rows`/`match_log` rows pulled this season) are the price at pull time, not a point-in-time
historical record.
