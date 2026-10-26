# Machine-learning models

Stage 5 asked whether machine-learned models can predict FPL points better than the [component model](prediction-model.md),
and kept whatever won. The answer is a modest yes: replacing the component model's minutes estimate with a learned one
is a small, consistent gain at every horizon, and the heavier models add little on top. The shipped default is
**ml-minutes**: the component model with LightGBM supplying the minutes probabilities.

Code: `features/ml_features.py`, `prediction/minutes_model.py`, `prediction/ml_points.py`, `prediction/ml_gru.py`,
`prediction/stacker.py`, `prediction/walkforward.py`, `prediction/ml_report.py`. The full comparison report is written
to `data/processed/ml_report.txt`.

## Goal

The component model's weakest part is minutes: whether a player starts, plays 60 minutes or is rotated. The aim was to
learn minutes and points from data, test every candidate on past seasons without leakage, and replace the component
model only if a candidate beat it by a rule fixed in advance.

## Data and new inputs

The match table gained seven columns per player and match: BPS (bonus points system score), the ICT components
(influence, creativity, threat), and the player's ownership and transfers in and out. The past-season archive was
re-pulled for them and the current season reads them from `element-summary`.

**Market timing check.** Ownership and transfers are only usable if the value recorded for a gameweek is the value at
the deadline, not one that already reflects the matches. The check compared each gameweek's figure with the same-week
and next-week values: in every season the same-week correlation was 0.92 to 0.94 against 0.29 to 0.43 for the next
week, so the same-gameweek values are used as deadline values (`MARKET_LAG = 0`).

One archive fix came out of the work: the 2025/26 archive repeated 10 rows exactly (20 duplicate rows in the match
table), and they are now dropped at source.

## Features

One row per player, upcoming fixture and deadline, for horizons 0 to 4 (0 is the next gameweek). Every feature uses
only matches that kicked off before the deadline's cutoff, the same rule as the component model, and a test checks
that removing future rows leaves the features unchanged.

- **Form:** rolling per-match and per-90 values over the last 1, 3, 5, 10 and 38 matches: minutes, starts, points,
  xG, xA, BPS, ICT, defensive contribution, saves, goals conceded.
- **Minutes pattern:** consecutive starts, days since last appearance, whether he played last match, new to the
  league or club, matches so far.
- **Fixture:** home or away, horizon, days ahead, fixtures in the gameweek, and the expected goals of the team and the
  opponent from the Poisson team ratings.
- **Player:** price, position, ownership and net transfers.
- **Scoring rules:** defensive contribution points only exist from 2025/26, so a `dc_rules` flag tells the models
  whether the season being predicted has them. Training targets are re-scored under the rules of the season
  being predicted.

New arrivals from outside the Premier League have no data before they arrive; the models learn their first weeks
from price, ownership, transfers, flags and the number of matches played. No pre-league data is used.

## Minutes model

A three-class model of how many minutes a player plays in a fixture: none, 1 to 59, or 60 or more. Three candidates
were compared by log loss on the same rows (lower is better):

| Model | 2023/24 | 2024/25 | 2025/26 |
|---|---|---|---|
| Component model's own | 0.6132 | 0.6424 | 0.5931 |
| LightGBM | 0.5474 | 0.5705 | 0.5167 |
| Keras MLP | 0.5549 | 0.5783 | 0.5249 |

LightGBM wins in every season. The calibration table shows why the component model loses: it is overconfident at the
top. In its highest decile of "plays 60 minutes or more" it predicts about 92% and 83% is observed; LightGBM predicts
86% and observes 85%, and is within about a point of the observed rate in every decile.

## Points if he plays, and how expected points combine

The points models predict a player's points in a fixture given the minutes class, as a function of the same features.
Expected points are then

```
E[points] = P(1-59 minutes) x points(1-59) + P(60+ minutes) x points(60+)
```

with the probabilities from the minutes model, summed over a gameweek's fixtures as in the component model. The models:

- **LightGBM** (regression, squared error).
- **Keras MLP** on the PyTorch backend (TensorFlow has no Python 3.14 wheel): standardised numerics, embeddings for
  team, opponent and position, dense layers with dropout, early stopping, the mean of three seeds.
- **ml-minutes:** not a new points model but the component model with its minutes probabilities replaced by the
  LightGBM ones. The component's goal, assist and clean-sheet machinery is untouched.

## GRU trial

A recurrent model over each player's last 10 matches plus the static features, tried as a possible improvement on the
MLP. Under the fixed rule it beat the MLP (J 1.7449 against 1.7518; horizon-0 RMSE difference -0.0028 [-0.0048,
-0.0010]), but it does not beat the simpler models, so it is not shipped. It is still logged live.

## Stacker

A non-negative linear blend, fitted separately for each position, of three members: ml-minutes, LightGBM and the GRU.
Weights are refitted at every deadline using only predictions that were out of sample and whose matches were already
played. They drift, but in broad terms defenders lean almost entirely on ml-minutes (about 0.92 to 0.95 at the end of
each season), while forwards put more weight on LightGBM and the GRU.

## Walk-forward design

- **Test seasons:** 2023/24, 2024/25 and 2025/26. 2022/23 is training only.
- **Expanding window:** models are retrained at the start of each season and every 4 gameweeks (gameweeks 1, 5, 9 and
  so on) using only rows whose fixture kicked off before the deadline. Between retrains a model is up to 3
  gameweeks stale, as it would be live.
- **Tuning:** the few hyperparameters (tree size for LightGBM, layer sizes and dropout for Keras, units and epochs for
  the GRU) were tuned on 2023/24 only, then frozen. 2023/24 is therefore in-sample for the ML settings, and the
  headline results use 2024/25 and 2025/26 ("held-out").
- **Metrics:** the same as the component model: RMSE, MAE, rank correlation (rho) and the points chosen by a simple
  decision rule, at horizons 0 to 4, with gameweek-block bootstrap intervals.

## The selection rule and result

The rule was fixed before the results. A candidate replaces the component model only if (a) its horizon-weighted
objective J (weights 0.85 per horizon, against rebuilt FPL form; lower is better) is lower, and (b) the 95% interval of
its horizon-0 RMSE difference from the component model lies wholly below zero. Among passing candidates the simplest
wins, in this order: ml-minutes, LightGBM, stacked, Keras.

Held-out (2024/25 and 2025/26), horizon 0, RMSE difference from the component model (negative is better):

| Model | J | RMSE difference [95% interval] | Passes |
|---|---|---|---|
| Component | 1.7490 | | |
| ml-minutes | 1.7209 | -0.0300 [-0.0400, -0.0213] | yes |
| LightGBM | 1.7309 | -0.0229 [-0.0347, -0.0126] | yes |
| Stacked | 1.7212 | -0.0347 [-0.0441, -0.0265] | yes |
| GRU | 1.7449 | -0.0216 [-0.0332, -0.0115] | yes |
| Keras MLP | not scored | -0.0188 [-0.0309, -0.0086] | |

**Choice: ml-minutes.** Mean per-gameweek RMSE at horizon 0 was 2.480 (component), 2.456 (ml-minutes) and 2.449
(stacked) in 2024/25, and 2.632, 2.596 and 2.594 in 2025/26; rebuilt form scored 2.782 and 2.973. Rank correlation
rose from 0.544 (component) to 0.566 (ml-minutes) over the two seasons. The points chosen by the decision rule did not
differ between the component model and the candidates beyond noise (about 63 per gameweek for all four). MAE at
horizon 0 is slightly worse for ml-minutes (+0.008), which is why the rule uses RMSE: the learned models predict the
mean better and the median slightly worse.

The gain is small in absolute terms, about 0.03 points of RMSE, but it is consistent. ml-minutes minus component, by
horizon:

| Horizon | RMSE difference [95% interval] |
|---|---|
| 0 | -0.0300 [-0.0400, -0.0213] |
| 1 | -0.0176 [-0.0267, -0.0101] |
| 2 | -0.0119 [-0.0183, -0.0057] |
| 3 | -0.0079 [-0.0137, -0.0023] |
| 4 | -0.0072 [-0.0144, -0.0002] |

The stacker is better than ml-minutes by a further -0.0047 [-0.0084, -0.0011] RMSE at horizon 0, but its MAE is worse
(+0.023), so the simpler model is kept.

## Breakdowns

Horizon 0, held-out seasons, ml-minutes against the component model (RMSE):

- **Position:** better in every position (defenders 2.508 against 2.537, midfielders 2.452 against 2.482, forwards
  2.763 against 2.787, goalkeepers 2.440 against 2.481).
- **Price band:** better in the three cheaper bands; in the dearest band (313 player-gameweeks) it is slightly worse
  (4.215 against 4.164), a small sample.
- **Hauls:** every model, the component model included, under-predicts players who score 5 or more points by about
  4.7 points on average. This is the nature of a mean forecast of a rare event, and a reason to consider
  haul-probability models later.
- **New arrivals:** about equal (1.356 against 1.342, 906 player-gameweeks).
- **Early season (gameweeks 1 to 5):** better (2.435 against 2.467), so the learned minutes help when there is little
  history.

## Decision

ml-minutes is the default. It is the simplest candidate that passes the rule, it is better than the component model at
every horizon, it is cheap to retrain, and it keeps the interpretable component model for everything but minutes. The
heavier models gain little extra and add training cost. Every candidate (component, ml-minutes, LightGBM, Keras, GRU
and stacked) is logged at each live deadline, so the 2026/27 season is a forward test that none of the settings have
seen.

## How to use it

```
python -m prediction.cli --model ml-minutes
python -m optimisation.cli --entry <team ID> --model stacked
```

`--model` accepts component, ml-minutes, lgbm, keras, gru and stacked; the default is ml-minutes. Live models are
retrained when the latest finished gameweek is 4 or more past the saved models, as in the validation, inside the first
pre-deadline run that needs it (about 16 minutes). A candidate that fails to train is skipped with a warning and never
blocks the default. Dependencies added: `lightgbm`, `keras` and `torch`.

The planner can be re-tuned on these predictions with `python -m evaluation.tune --model ml-minutes --fresh`. That
replays only 2023/24 to 2025/26, the seasons with walk-forward predictions, and writes its settings to a separate file
(`plan_params_ml-minutes`). It takes hours and has not been run yet.

## Limits and caveats

- **2023/24 is in-sample for the ML settings.** Headline figures use the two later seasons. Including 2023/24 changes
  the rule's choice to stacked (J 1.7265 against ml-minutes 1.7274, a negligible difference), which shows how close
  the top candidates are.
- **The component model's own parameters were tuned on all four seasons**, including the test seasons, which flatters
  it relative to the ML models.
- **Keras and GRU settings were tuned before two preprocessing fixes** (binary inputs left unstandardised, and the
  defensive-contribution features, which do not exist before 2025/26, no longer inflating 2025/26 inputs). They may be
  slightly understated.
- **No historical injury flags.** The validation had no record of what FPL's chance-of-playing flag said at each past
  deadline; live use applies the flags on top of the minutes model, so live behaviour is not exactly what was tested.
- **No pre-Premier-League data** for new arrivals.
- **Live net transfers are partial** before the deadline, since the figures keep moving until it passes; the models
  were trained on deadline values.
- **Small differences.** The gains are real but small, and the benchmark is rebuilt FPL form, not FPL's own forecast;
  the live `ep_next` record is the better test once enough gameweeks are played.
