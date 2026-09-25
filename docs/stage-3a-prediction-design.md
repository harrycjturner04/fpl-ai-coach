# Stage 3a design: component prediction model and backtest

Status: approved design, 2026-09-25. Implementation plan to follow.

## 1. Problem

The Stage 2 optimiser needs a score per player. Its placeholder score (`0.7 × ep_next + 0.3 × form`) is effectively
recent form: FPL's `ep_next` equals `form` for about 92% of players, so it ignores fixtures entirely.

Stage 3a replaces it with a model of **expected FPL points per player, per fixture, for the next five gameweeks**, and a
walk-forward backtest that proves whether the model beats FPL's own expected points.

Stage 3 is split in two:

- **3a (this document):** optimiser CLI test, the prediction model, and the backtest.
- **3b (later):** the multi-week optimiser, extending the free-transfer bank already in the ILP.

## 2. Decisions

| Topic | Decision |
|---|---|
| Scope | 3a = CLI test + prediction + backtest; 3b = multi-week optimiser |
| Data for tuning/testing | Past seasons from the public archive (vaastav/Fantasy-Premier-League) plus the current season |
| Seasons | 2022/23 to 2025/26: identical columns, including xG/xA, starts, position, team and FPL's `xP` |
| Model | Component model (team ratings from xG, player shares, points from FPL's scoring rules) |
| Minutes | Recency-weighted rates with shrinkage (logistic model deferred to Stage 5) |
| Tuning objective | MAE and within-position rank correlation, relative to FPL's `xP` |
| Holdout | 2025/26 is locked until one final evaluation |
| Current-season data | FPL `event/{gw}/live` (one request per gameweek) |
| Spec location | `docs/` in the repository |

### Data facts that shaped the design

- This season has 5 gameweeks: too few to tune on, hence the archive.
- FPL's team strength fields are blank this season (all 0 or None), so team strength is our own rating.
- Playing time dominates: 3.88 points per appearance at 60+ minutes versus 1.21 below 60.
- The archive has stable keys across seasons: player `code` and team `code`.
- The archive's licence is unspecified: use it for analysis only, never republish it (`data/` is not committed).
- Scoring rules differ by season: defensive contribution points exist from 2025/26 only.

## 3. Architecture

```
ingestion/archive.py           one-off download of 2022/23 to 2025/26 (re-run when a season ends)
ingestion (event_live)         current season, one request per finished gameweek
    -> data/processed/match_log.parquet
       season, gameweek, kickoff, player_code, team_code, opponent_code, was_home, position,
       minutes, starts, goals, assists, xG, xA, xGC, clean_sheet, goals_conceded, saves, bonus,
       defensive_contribution, yellow, red, points, price, xP (archive only)

features/team_ratings.py       match log before a cutoff -> attack/defence rating per team
features/player_rates.py       match log before a cutoff -> p60, psub, shrunk per-90 rates
prediction/scoring.py          FPL points per component, per position, per season
prediction/component_model.py  features + fixtures + flags -> expected points per player,
                               per fixture, for the next 5 gameweeks (components kept)
prediction/backtest.py         walk-forward replay and metrics
prediction/tune.py             coordinate-descent parameter search
prediction/cli.py              live predictions table
    -> data/processed/predictions.parquet, data/processed/model_params.json

optimisation/cli.py            --scorer component (default) | placeholder
```

**One timeline rule:** every feature function takes a cutoff time and reads only matches that kicked off before it.
Live predictions use "now"; the backtest uses each past gameweek's first kickoff. Same code path for both.

Storage: `match_log.parquet` keeps only the columns above (a few MB for all seasons).

## 4. The model

Core equation, for player *i* in gameweek *g*:

```
E[pts_i,g] = Σ over fixtures f of i's team in g of:  components(i, f)
```

A double gameweek contributes two fixtures; a blank contributes none.

### 4.1 Team ratings

For a match where H hosts A:

```
λ_H = exp(μ + γ + a_H − d_A)
λ_A = exp(μ     + a_A − d_H)
```

- `a_T`, `d_T`: attack and defence ratings. `μ`: league baseline. `γ`: home advantage. All fitted.
- Fitted by weighted Poisson maximum likelihood with a ridge penalty `ρ·Σ(a² + d²)` (`scipy.optimize`).
- Target: `α·xG + (1−α)·goals`. Tunable α, default 1.0 (pure xG).
- Match weight halves every `h_team` days (tunable). Previous-season matches carry an extra fade (tunable).
- Promoted teams shrink towards the average rating of past promoted teams instead of 0.

### 4.2 Minutes

```
p60  = shrink( Σ w_k·[mins_k ≥ 60] / Σ w_k ),   w_k = 0.5^(age_k / h_min)
psub = shrink( Σ w_k·[0 < mins_k < 60] / Σ w_k )
E[m] = p60·m60 + psub·msub
```

- `m60`, `msub`: the player's typical minutes in each case, shrunk towards about 87 and 25.
- Shrinkage in general: `shrunk = (Σ w·x + κ·prior) / (Σ w + κ)`, with tunable `κ`.
- Minutes prior: the average `p60`/`psub` for the player's position and price band, computed before the cutoff.
- Injury flag: next gameweek multiplied by `c = chance_of_playing / 100`; week k ahead uses
  `c_k = 1 − (1−c)·δ^k` (tunable δ).

### 4.3 Attacking rates in team context

```
r_xG,i = shrink( Σ w_k·xG_k / Σ w_k·(mins_k/90) ) ÷ (mean λ of i's team in those matches)
```

Multiplied back by the current team's `λ` per fixture, so a player moving to a stronger team gains expected goals. xA
likewise. Prior for shrinkage: average for the player's position and price band (<£5.5m, £5.5–7.5m, £7.5–10m, >£10m).

### 4.4 Components per fixture

Player *i* (position *pos*), team *T*, opponent *O*:

| Component | Formula |
|---|---|
| Appearance | `2·p60 + 1·psub` |
| Goals | `pts_goal(pos) · r_xG,i · λ_T · E[m]/90` (6 GK/DEF, 5 MID, 4 FWD) |
| Assists | `3 · r_xA,i · λ_T · E[m]/90` |
| Clean sheet | `pts_cs(pos) · p60 · e^(−λ_O)` (4 GK/DEF, 1 MID) |
| Goals conceded | `−p60 · E[⌊G/2⌋]`, `G ~ Poisson(λ_O)` (GK/DEF) |
| Saves | `p60 · E[⌊S/3⌋]`, `S ~ Poisson(save_rate_i · λ_O / λ̄)` (GK) |
| Bonus | shrunk bonus per 90 × `E[m]/90` |
| Defensive contribution | `2 · p60 · p_dc,i` (from 2025/26; threshold 10 for DEF, 12 for MID/FWD) |
| Discipline | `−(1·yellow_rate + 3·red_rate) · E[m]/90` |

Points values come from `prediction/scoring.py` per season.

Known approximations: clean-sheet and conceded points assume a 60+ minute player is on for the whole match; goals and
assists are treated as independent; bonus does not react to the fixture. The backtest measures whether these cost
accuracy.

## 5. Backtest and tuning

### 5.1 Replay

- Every gameweek of 2022/23 to 2025/26; 2022/23 GW1–5 are warm-up only. GW1–5 of other seasons reported separately.
- Cutoff: the gameweek's first kickoff.
- Predictions for g to g+4; tuning uses g (next week).
- Actual: the player's total points that gameweek (both fixtures in a double).
- Scored players: those with minutes in any of their last 5 matches before the cutoff.

### 5.2 Metrics (per horizon, against FPL's `xP` and a naive last-5 average)

- MAE and RMSE.
- Spearman rank correlation within each position, averaged over gameweeks.
- Decision value: the real Stage 2 optimiser picks a from-scratch squad from each model's predictions at that week's
  prices; score the XI and captain on actual points.
- Calibration: 10 groups by predicted points, mean predicted versus mean actual.

### 5.3 Tuning

```
J = MAE_model / MAE_xP + (1 − ρ_model) / (1 − ρ_xP)
```

J = 2 means level with FPL's `xP`; lower is better. Search by coordinate descent (one parameter at a time over a small
grid, repeated passes until no improvement), with team-side and player-side features cached separately. Tuned on
2022/23 to 2024/25; 2025/26 evaluated once with frozen parameters. Output: `model_params.json`.

### 5.4 Leakage safeguards

1. Team and price come from each match row, never the season-end player file (which leaks mid-season transfers).
2. Every average and prior is computed from data before the cutoff.
3. 2025/26 is never used for tuning.
4. The archive holds the final fixture schedule (including later-arranged double gameweeks); this slightly flatters
   predictions more than one week ahead and is reported as a limitation. 3b will consider reconstructing the schedule
   as known at the time.
5. Historical injury flags are unavailable, so the backtest omits them; live predictions use them, so live accuracy
   should be at least as good as backtested.
6. FPL's `xP` is a benchmark only, never a model input.

Enforced by a test: features computed at a cutoff are identical with or without made-up future matches appended.

## 6. Live use

- `event_live(gw)` fetches each finished gameweek once, then only the current gameweek.
- `python -m prediction.cli [--position MID] [--top 20]` shows five-week predictions with components and `p60`; saves
  `predictions.parquet`. Uses defaults with a warning until `model_params.json` exists.
- Players without history use position/price-band and cautious minutes priors, flagged `no history`.
- `optimisation.cli` uses next gameweek's predicted total by default (`--scorer placeholder` for Stage 2's score).
  The ILP constraints do not change in 3a.
- New dependency: `scipy`.

## 7. Testing

- Leakage invariance for team ratings, player rates and priors.
- Team-rating parameter recovery from simulated Poisson seasons with known ratings.
- Poisson helpers (`e^(−λ)`, `E[⌊G/2⌋]`, `E[⌊S/3⌋]`) against brute-force sums.
- Minutes weighting, shrinkage limits, flag fade.
- One player-fixture computed by hand; double gameweek = sum of fixtures; blank = 0; defensive contribution only from
  2025/26.
- Metric functions on data with known answers; decision value on a small pool.
- Archive and `event_live` transforms on small sample files; `live`-marked schema checks for both sources.
- Offline tests for `prediction.cli` and `optimisation.cli`.

## 8. Build order

1. `optimisation/cli.py` offline test.
2. Archive ingestion, `event_live`, `match_log`.
3. Backtest harness with the two benchmarks only (numbers to beat, before any model).
4. Team ratings, then clean-sheet and conceded components, then backtest checkpoint.
5. Minutes and appearance, then checkpoint.
6. Attacking rates, goals and assists, then checkpoint.
7. Saves, bonus, defensive contribution, discipline, then checkpoint.
8. Tuning, then the single 2025/26 evaluation, then the results table.
9. Live prediction CLI, optimiser switched to the new score, documentation.

At each checkpoint the metric changes against FPL's `xP` are reported; a component that does not help is discussed
before continuing.

## 9. Error handling

- Archive download failure: clear message; live data unaffected.
- Missing `model_params.json`: documented defaults with a warning.
- Rating fit fails to converge: stop with a message.
- Players with no history: flagged in output.
