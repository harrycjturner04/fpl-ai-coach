# FPL AI Coach

An AI-assisted Fantasy Premier League coach. It combines a statistical/ML prediction layer, an integer linear programming (ILP) squad optimiser, and an LLM reasoning layer to recommend transfers, starting XI and captaincy; with explanations.

## Architecture

Five layers, each with a single responsibility; only clean, structured output crosses a layer boundary.

1. **Ingestion**: official FPL API (players, fixtures, per-player history, a manager's history/transfers/picks), later squad-screenshot parsing and injury/news sources.
2. **Feature engineering**: rolling form, per-90 rates, fixture difficulty, minutes reliability, set-piece duty, team strength.
3. **Prediction**: expected points per player for the next 1–5 gameweeks, built component-by-component from team attack/defence ratings, player minutes/start probability and per-90 rates, combined through FPL's own scoring rules; next, gradient-boosted trees validated walk-forward by gameweek. Minutes/start probability is modelled separately.
4. **Optimisation**: a PuLP integer linear program that plans five gameweeks at once, covering budget, 2/5/5/3 squad, max 3 per club, valid formation, captaincy, free transfers and the −4 transfer hit.
5. **Reasoning**: an LLM that turns news into structured risk flags, reviews the optimiser's output, plans chip timing, and writes the recommendation, calling the optimiser as a tool to test scenarios.

## Status

- [x] Stage 1: Ingestion skeleton (FPL API → raw JSON snapshots + Parquet tables)
- [x] Stage 2: ILP optimiser on current stats (from-scratch squads and transfer plans)
- [x] Stage 3a: Component prediction model (appearance, goals, assists, clean sheets, bonus, etc.), live and feeding the optimiser by default
- [x] Stage 3b: Multi-week optimiser (five-week plan, chips as what-ifs), tested by replaying four past seasons
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
python -m optimisation.cli --entry 1234567   # this week's transfers for your team, plus a provisional 5-week plan
python -m optimisation.cli --entry 1234567 --horizon 3 --discount 0.9   # override the shipped settings
python -m optimisation.cli --entry 1234567 --chip bboost:12             # what-if: a chip in a gameweek
                                             # (chips: wildcard, freehit, bboost, 3xc; repeatable)

python -m evaluation.tune --workers 16       # replay four seasons, tune the planning settings (several hours)
python scripts/make_charts.py                # redraw the charts in docs/img/

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

A single integer linear program (PuLP + HiGHS) plans **five gameweeks at once**: squad, transfers, starting XI and
captain for every week, maximising discounted expected points (starters + captain bonus + 0.1 × bench − 4 × hits)
plus a measured value for free transfers left over at the end, subject to budget, 2/5/5/3 composition, max 3 per club,
a valid formation, FPL's free-transfer rollover (+1 a week, cap 5) and at most two hits a week. Only week one is acted
on; the plan is re-solved every week with new data. Chips are what-ifs: name one (`--chip bboost:12`) and the plan
applies it; choosing when to play chips is left to the reasoning layer. Solves are exact (proven optimum): 47 to 50 s
for a five-week plan on 562 players, under a second for one week.

Correctness is enforced three ways: an independent checker (`optimisation/validate.py`) re-checks every rule in every
planned week and between weeks after each solve; hand-built cases with known answers; and randomised cross-checks
against brute-force enumeration of every plan on a scaled-down game over two and three weeks. HiGHS and CBC are also
checked against each other.

Why an ILP: expected squad points are a sum of player expected points, so the problem is linear and the solver returns
the proven optimum. Greedy, knapsack DP, genetic algorithms and RL were considered and rejected (not exact, don't scale
to these constraints, or need a season simulator).

Settings (discount 0.7, bench weight 0.1, the leftover-transfer values) were tuned by replaying past seasons and ship in
`optimisation/plan_params.json`. Players are scored by the component prediction model; pass `--scorer placeholder` to
fall back to the Stage 2 ep_next/form blend (one week only). Full method: [docs/optimiser.md](docs/optimiser.md).

## Planning results

Each of four past seasons was replayed as a simulated manager: start from a £100m squad, then every gameweek predict,
plan, make week one's transfers and score real points with FPL's automatic substitutions (146 gameweeks, no chips).
Settings were validated leave-one-season-out: each season is scored with the settings that did best on the other three.

| Policy | 2022/23 | 2023/24 | 2024/25 | 2025/26 | Total |
|---|---|---|---|---|---|
| No transfers | 1,656 | 1,714 | 1,840 | 1,129 | 6,339 |
| Single-week optimiser | 2,004 | 1,944 | 2,144 | 2,013 | 8,105 |
| Simple look-ahead (one-week solve on discounted 5-week totals) | 1,947 | 2,238 | 2,217 | 2,171 | 8,573 |
| Multi-week, held out (settings chosen on the other three seasons) | 1,970 | 2,080 | 2,213 | 2,139 | 8,402 |
| Multi-week, shipped settings (in-sample) | 2,137 | 2,251 | 2,213 | 2,139 | 8,740 |

Held out, the multi-week model scored +2.03 points per gameweek more than single-week planning (95% interval −0.10 to
+4.33, so it narrowly includes zero); the simple look-ahead scored +3.21 (+0.55 to +5.99). Looking ahead helps, by about
2 to 3 points a gameweek, but the full multi-week model and the simple look-ahead are not distinguishable in four
seasons of replay (held-out difference −1.17, interval −3.37 to +1.09). Differences between settings are within the
replay's noise, so the shipped settings are the best estimate rather than a proven optimum. The true test is the live
2026/27 season, scored forward against FPL's recorded `ep_next`. Details and limits: [docs/validation.md](docs/validation.md).

## Prediction results

Stage 3a holdout: walk-forward backtest on the held-out 2025/26 season (parameters tuned on 2022/23 to 2024/25 only; 2025/26 was
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
matches 5 days on next-gameweek RMSE and beats it at every later horizon. The Stage 3a model therefore shipped with
10 days. The holdout was not re-run, to keep it a single, untouched evaluation; Stage 3b replaced the single memory
with a horizon-dependent blend (below).

Backtest fix after the holdout: code review found that players whose club had no fixture in a gameweek (a blank)
were left out of that gameweek's snapshot, so they were not predicted for the following weeks either. They are now
included. Blanks affect a few gameweeks per season (two in 2025/26); on the tuning seasons the fix shifts every
model's metrics slightly and leaves the comparison unchanged (component minus form: RMSE -0.31, rank correlation
+0.07, both intervals excluding zero). The holdout figures above predate this fix.

Stage 3b re-tune: the minutes model now blends a short memory (5 days) with a long one (60 days) according to how far
ahead the fixture is, and parameters were re-tuned on all four seasons (2025/26 included) weighting all five horizons.
Next-gameweek accuracy barely changed and later horizons improved (RMSE five gameweeks ahead 2.706 → 2.663, in-sample).
These figures are in-sample; the next untouched test is the live 2026/27 season. See
[docs/prediction-model.md](docs/prediction-model.md).

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
data/processed/prediction_log.parquet the model's own predictions, recorded before each deadline (forward test)
data/processed/plan_log.parquet       the optimiser's recommendation for a team, recorded before each deadline
                                      (latest per season, gameweek and team)
data/processed/replay_predictions.parquet  predictions for every replayed gameweek (evaluation/tune.py)
data/processed/replay_results.parquet one row per replayed gameweek and setting: points, hits, transfers
data/processed/replay_report.txt      the replay's season totals, leave-one-season-out and hits report
data/processed/leftover_table.json    measured value of leftover free transfers, every fixed-point pass
data/processed/plan_params.json       optimiser settings from a local re-tune (evaluation/tune.py); takes
                                      priority over the copy shipped at optimisation/plan_params.json
data/processed/game_rules.json        budget, squad composition, club cap, FT cap, sell-on fee
data/processed/metadata.json          when data and each team were last pulled (drives auto-refresh)
data/processed/manager_state_<id>.json
prediction/model_params.json          tuned parameters shipped in the repo, so a fresh clone uses them
                                      without a local re-tune
optimisation/plan_params.json         tuned optimiser settings shipped in the repo, likewise
```

`data/` is not committed; run the CLI to regenerate it. Current-season player prices (`players.price`, and
`current_season_rows`/`match_log` rows pulled this season) are the price at pull time, not a point-in-time
historical record.
