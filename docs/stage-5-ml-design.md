# Stage 5: machine-learning prediction, design

## Context

Stages 3a/3b are merged: a component model (Poisson team xG ratings, minutes model, per-90 rates, FPL scoring rules) feeds a multi-week ILP planner. Next-GW RMSE 2.574 (in-sample); holdout 2025/26 beat rebuilt form by 0.35 RMSE. Its weakest part is minutes, the same weakness OpenFPL (arXiv 2508.09992) showed vs commercial models. Stage 5 adds machine-learned models, compared honestly with the component model, and ships whatever wins. The user wants the network in Keras: TensorFlow has no Python 3.14 wheel, so Keras 3 runs on the PyTorch backend (torch 2.14.1, keras 3.15.1 available); no compromise needed.

Agreed with the user (2026-10-14): goal = GBM and Keras honestly compared; scope = ML minutes model, ML points models, stacking (predicting ranges/haul probability deferred but designed for); new inputs = BPS, ICT, ownership, transfers; judged by walk-forward 2023/24-2025/26 plus live 2026/27 forward test; approach A; Keras MLP first, GRU as a trial.

## 1. Data, features, targets

- **Ingestion**: add `bps`, `influence`, `creativity`, `threat`, `selected`, `transfers_in`, `transfers_out` to `ingestion/transform.py:MATCH_COLUMNS`; archive (`ingestion/archive.py`, vaastav `merged_gw.csv` has them) re-pulled; current season from `player_history` (already has them).
- **Leakage check (checkpoint 1)**: `ingestion.cli` starts logging bootstrap `selected_by_percent`, `transfers_in_event`, `transfers_out_event` at each pre-deadline run; once a GW is played, compare them with that GW's `player_history` values to learn whether the per-GW figures are as of the deadline. If they match, use them for the GW being predicted; if not (or too few GWs logged by checkpoint 1), use only the previous GW's values, which are always known at the deadline.
- **Feature table** `features/ml_features.py`: one row per player x upcoming fixture x deadline, horizons 0-4 (0 = next GW, as in the existing code), built only from `prediction/backtest.py:snapshot` history (kickoff < cutoff). Features: rolling per-match and per-90 over last 1/3/5/10/38 matches (minutes, starts, points, xG, xA, BPS, I/C/T, defcon, saves, goals conceded); minutes pattern (consecutive starts, days since last appearance, played last match, new to league/club, matches so far); fixture (home, horizon, days ahead, fixtures in GW, lam_team/lam_opp from `features/team_ratings.py`); player (price, position, ownership, net transfers). Cached as Parquet with an input fingerprint.
- **New-to-league players**: learned from four seasons of arrivals via price, ownership/transfers, flag and match count; the component model's position x price-band priors (`features/player_rates.py`) remain inside it. Pre-PL data (FBref) out of scope; qualitative news is Stage 6.
- **Targets**: minutes class (0 / 1-59 / 60+); points re-scored under 2026/27 rules from raw stats via `prediction/scoring.py:points_from_stats`.

## 2. Models and how they connect

```
history -> team ratings (3a) ------------------+--------------------+
        -> ML minutes model (5, shared) -------+                    |
                                               v                    v
                   component model (3a) with ML minutes    LightGBM / Keras points-if-plays
                                               +--> stacker <-------+
                                                      v
                                     expected points per player per GW -> planner (3b, unchanged)
```

- **Minutes** (`prediction/minutes_model.py`): LightGBM multiclass and Keras MLP softmax; pick by walk-forward log loss. Live: FPL `chance_of_playing` applied on top with the existing `flag_fade`.
- **Points if he plays** (`prediction/ml_points.py`): minutes class is an input; expected points = P(1-59)*pts(1-59) + P(60+)*pts(60+). LightGBM (L2). Keras MLP: standardised numerics + embeddings (team, opponent, position), 2-3 dense layers, dropout, early stopping on the latest pre-deadline data, `KERAS_BACKEND=torch`, 3-seed average. GRU trial over the last 10 matches plus static features.
- **Component + ML minutes**: `prediction/component_model.py:predict` gains an optional minutes input replacing its p60/psub.
- **Stacker** (`prediction/stacker.py`): non-negative linear blend per position of component(+ML minutes), LightGBM, Keras; fitted only on out-of-sample predictions from earlier GWs.
- **Walk-forward** (`prediction/walkforward.py`): test seasons 2023/24-2025/26 (2022/23 training only); expanding window, retrain at season start and every 4 GWs using only rows whose fixture kicked off before the deadline; models cached in `data/models/<deadline>/`. Small hyperparameter grid tuned on 2023/24 only, then frozen; 2023/24 is therefore in-sample for the ML models, and their headline results are reported on 2024/25-2025/26. Fixed seeds.
- **Dependencies**: `lightgbm`, `keras`, `torch` (CPU).

## 3. Evaluation and choosing the winner

- Every candidate is a `Predictor` in `prediction/backtest.py:run_backtest`; metrics `score_metrics`, `decision_points`, horizons 0-4; `paired_bootstrap` vs component.
- Minutes: log loss and calibration table vs component's p60/psub-implied probabilities.
- Rule (fixed in advance): main score = J weighted 0.85^h (as 3b). To replace the component model a candidate must beat it on J and its next-GW RMSE difference CI must exclude zero. Within CIs, ship the simpler (component+ML minutes < LightGBM < stacker < Keras). GRU kept only if it beats the MLP by the same test. If nothing passes, the component model stays and the write-up says so.
- Breakdowns: position, price band, new-to-league, GW1-5, zeros vs 5+ pointers.
- Live 2026/27: `prediction_log` records every candidate each deadline; scored against component and FPL `ep_next`.
- After shipping: re-tune 3b settings (`evaluation/tune.py`) with the new predictions and re-run multi-week vs look-ahead.

## 4. Code, live use, tests, docs

- `optimisation/` unchanged. `prediction.cli` / `optimisation.cli` gain `--model {component,ml-minutes,stacked,...}`; default switches only after checkpoint 3. Models retrain automatically before a deadline when a new GW has finished.
- Tests: leakage (removing future rows leaves features unchanged); re-scored targets match archive points under each season's rules; shapes; tiny synthetic training run (Keras 2 epochs); stacker weights non-negative; same seed gives same predictions; live test for new API columns.
- Docs: spec `docs/stage-5-ml-design.md` (committed); plan `docs/plans/` (local); after the build `docs/ml-model.md` and README status; CLAUDE.md sections 3, 5, 7 and changelog.
- Built subagent-driven.

## Checkpoints (user review)

1. Data: new columns, leakage verdict, feature table.
2. Minutes: log loss and calibration, LightGBM vs Keras vs component.
3. Points models and stacker: walk-forward results with CIs; ship decision.
4. Live switch-over and all candidates logging; then 3b re-tune.

## Verification

- `pytest` (offline) and `pytest -m live` pass.
- `python -m prediction.backtest --seasons 2023-24,2024-25,2025-26 --models component,ml-minutes,lgbm,keras,stacked` reproduces checkpoint tables.
- `python -m optimisation.cli --entry 8298351 --model <shipped>` produces a legal plan (validator runs automatically).

