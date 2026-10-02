# Stage 3b design: multi-week optimiser

Status: design agreed 2026-10-02, not yet built.

## 1. Problem

The Stage 2 optimiser plans one gameweek. Stage 3a predicts expected points `p[i,t]` for every player `i` in each
of the next five gameweeks `t`. Used one week at a time, those predictions cannot answer the questions that decide
a season: whether to buy a player a week early for a double gameweek, whether to keep a free transfer because a
better use is coming, whether a 4-point hit pays back over several weeks rather than one.

Stage 3b extends the optimiser to plan several gameweeks at once and proves, by replaying past seasons, that doing
so earns more real points than planning one week at a time.

## 2. Decisions

| Topic | Decision | Main alternative rejected |
|---|---|---|
| Formulation | Multi-period integer linear program on a rolling horizon | Single-week model on summed scores (cannot time transfers); search heuristics (no optimality proof) |
| Uncertainty | Expected values, with a discount on later weeks | Scenario-based stochastic program: deferred, see section 11 |
| Horizon | 5 weeks by default; 3 and 4 also evaluated | 8 weeks (needs predictions beyond what 3a was tested on) |
| Discount | `gamma` tuned in the season replay | Fixed 0.85 by convention |
| Leftover free transfers | Value measured from data | Hand-set constant |
| Prices | Frozen at today's values over the horizon | Forecasting price changes (unpublished algorithm) |
| Hits | True cost of 4, capped per week; margin added only on evidence | No hits in planned weeks |
| Chips | Applied when the caller names a chip and week; timing left to Stage 6 | Optimiser chooses timing (needs a value for holding a chip) |
| Minutes by horizon | Blend of short and long memory | First-order Markov chain (contradicted by the data, section 5.1) |
| Prediction tuning | Weighted over all five horizons; 2025/26 added to tuning data | Next gameweek only |
| Evaluation | Season replay with auto-substitutions; leave-one-season-out | Tune and score on the same seasons |
| Solver | HiGHS; CBC kept in tests as a cross-check | CBC only |
| Bench | One weight, tuned in the replay | A measured weight per bench slot (backlog) |

## 3. Architecture

- `optimisation/model.py`: the multi-period model. It still receives scores as opaque numbers, now a table of
  player by gameweek, and knows nothing about how they were produced.
- `optimisation/validate.py`: independent legality checker, extended to check every planned week and the
  transitions between weeks (squad carry-over, bank, free transfers, hits).
- `features/player_rates.py`, `prediction/component_model.py`, `prediction/tune.py`: minutes blend and the
  horizon-weighted tuning target.
- `evaluation/` (new package): season replay, auto-substitution scoring, measurement of the leftover-transfer
  value, tuning of the optimiser's settings. It uses both the prediction and optimisation layers, so it belongs in
  neither.
- `optimisation/cli.py`: prints this week's action and the provisional plan for the following weeks.

## 4. The model

### 4.1 Rolling horizon

Each week the model is solved for `H` gameweeks, only the first week's decisions are acted on, and the model is
solved again the following week with new data. The later weeks are a plan, not a commitment: their purpose is to
put the right value on this week's decision.

### 4.2 Variables

For each player `i` and week `t = 1..H`, all binary:

- `x[i,t]`: in the squad
- `y[i,t]`: in the starting XI
- `c[i,t]`: captain
- `b[i,t]`: bought before week `t`
- `s[i,t]`: sold before week `t`

For each week, integer or continuous:

- `m[t]`: money in the bank after week `t`'s transfers
- `f[t]`: free transfers available in week `t`
- `u[t]`: free transfers used
- `h[t]`: hits taken

### 4.3 Constraints

Squad carry-over, with `x[i,0]` the current squad:

`x[i,t] = x[i,t-1] + b[i,t] - s[i,t]` and `b[i,t] + s[i,t] <= 1`

Bank, with `m[0]` the current bank:

`m[t] = m[t-1] + sum_i sell_i * s[i,t] - sum_i price_i * b[i,t]` and `m[t] >= 0`

`price_i` is today's price. `sell_i` is today's selling price for a player already owned and `price_i` for anyone
bought inside the plan. Prices are held fixed over the horizon (section 2). A player owned today may be sold at
most once in the horizon, because a second sale after re-buying would be at a different price; the restriction
costs nothing in practice, since selling, re-buying and selling again within five weeks spends three transfers.

Transfers and hits, with `n[t] = sum_i b[i,t]`:

`u[t] <= f[t]`, `u[t] <= n[t]`, `h[t] = n[t] - u[t]`, `h[t] <= hit_cap`

Free-transfer rollover. FPL's rule is `f[t+1] = min(F, f[t] - u[t] + 1)` with `F = 5`. A minimum is not linear,
so it is written as two upper bounds:

`f[t+1] <= f[t] - u[t] + 1` and `f[t+1] <= F`

The solver raises `f[t+1]` to the true value on its own, because more free transfers can only help later. It also
uses free transfers before hits: saving a free transfer by paying for a hit could at best avoid one later hit at
an equal or smaller weight, or earn a leftover value that is required to be below the hit cost. The reported
values are recomputed from FPL's exact rule after the solve, and the validator checks them.

Every week, the Stage 2 rules apply unchanged: 15 players in the 2/5/5/3 composition, 11 starters in a valid
formation, at most three players per club, exactly one captain who starts. Limits are read from FPL's own data.

From-scratch mode (no current squad) is the same model with an empty `x[i,0]`, a bank equal to the budget, and
week 1's transfers free and unlimited.

### 4.4 Objective

Maximise

`sum_t gamma^(t-1) * [ sum_i p[i,t] * (y[i,t] + c[i,t] + w * (x[i,t] - y[i,t])) - (4 + delta) * h[t] ]`
`+ gamma^H * sum_k Delta_k * z_k - tiebreak * sum_{i,t} b[i,t]`

- `p[i,t]`: predicted points. A starter counts once, the captain twice, a bench player at weight `w`.
- `gamma`: discount on later weeks (4.5).
- `delta`: hit margin, zero unless the replay shows hits lose real points (6.3).
- `Delta_k`, `z_k`: value of free transfers left after week `H` (4.6).
- `tiebreak`: a tiny penalty so the model never makes a transfer that gains nothing.

With `H = 1` this is exactly the Stage 2 model, so its known-answer and brute-force tests stay as regression tests.

### 4.5 Why discount later weeks

Every gameweek's points count equally in FPL, so `gamma` is not a preference for points now. It corrects two
biases of planning on point estimates:

1. Selection bias. The solver takes the maximum over many noisy estimates, so the options it picks are
   disproportionately the overestimated ones. The noisier the estimates, the larger the bias, and predictions for
   later weeks are noisier (rank correlation falls from 0.52 for the next gameweek to 0.42 four weeks further on).
2. Plan revision. The model is re-solved every week, so a move planned for week 4 often never happens. A fixed
   plan overstates what its later weeks will deliver.

Both effects are empirical, so `gamma` is chosen by the replay (section 6) from 0.7, 0.8, 0.9 and 1.0.

### 4.6 Value of leftover free transfers

Without a value on transfers left after week `H`, the model spends every free transfer in its last planned week
on any small gain, which distorts the weeks before it. Inside the horizon no such value is needed: if keeping a
transfer until week 3 is worth more than using it now, the model sees that directly.

The value is measured. Let `W(f)` be the model's optimal objective from a given squad and bank when it starts
with `f` free transfers. One more transfer is worth

`Delta(f) = W(f+1) - W(f)`, for `f = 1..4`

Each `Delta(f)` is the average of that difference over sampled gameweeks of the replay. The best transfer is used
first, so the table decreases, which keeps the model linear: with `z_k = 1` when at least `k+1` transfers remain,
the term `sum_k Delta_k * z_k` needs no extra ordering constraints.

The definition is circular, since `W` contains the leftover value being measured. It is resolved by iteration:
start from zero, measure the table, put it into the model, measure again, and stop when it settles (at most three
passes). This is value iteration from dynamic programming on a single state variable.

The table is in predicted points and therefore inflated by the selection bias of 4.5. The replay scores the table
at scales 0, 0.5 and 1 on real points and keeps the best. Every `Delta_k` must stay below the hit cost.

### 4.7 Chips

The caller may name one chip for a given week; the model applies its effect and does not choose timing.

- Bench Boost in week `t`: the bench weight `w` becomes 1 for that week.
- Triple Captain in week `t`: the captain's extra term counts twice (three times his points in total).
- Wildcard in week `t`: that week's transfers are unlimited and free, `h[t] = 0`, and `f[t+1] = f[t]` (the bank of
  free transfers is kept and none is added, the rule verified in Stage 1).
- Free Hit in week `t`: a separate squad `q[i]` is chosen for that week only, with unlimited free transfers and a
  budget of the bank plus the selling value of the regular squad. The regular squad is untouched, so week `t+1`
  continues from week `t-1`'s squad and bank, and `f[t+1] = f[t]`. An owned player kept in the Free Hit squad
  counts at his selling price, not his higher buy price; this needs one linking variable per owned player.

Whether a chip is actually available is checked by the caller from the manager state built in Stage 1.

### 4.8 Solver and model size

About 700 players, five weeks and five binary variables each: roughly 17,500 binaries, against 2,100 today. The
model moves to the HiGHS solver (`highspy`), which PuLP drives through the same interface. CBC stays available and
is used in tests to check that both solvers reach the same optimum.

Trimming the player pool was agreed in its exact form: remove a player only when enough others in his position
cost no more and are predicted at least as many points in every week, so that removing him provably cannot change
the optimum. Working through the multi-week case shows the safe threshold is much higher than for one week (a
replacement must be free in every week the player would be held, across all the squads a plan passes through), so
the rule may remove little. HiGHS also performs this kind of reduction itself. The untrimmed model is therefore
timed first, and trimming is added only if needed. If it is needed and the exact rule is too weak, that choice
comes back to the project owner at the first checkpoint rather than being settled silently.

Live solves are always to proven optimality. Whether replay solves may stop within a small gap of the bound is
also a checkpoint decision, made with timings in hand.

## 5. Predictions further ahead

### 5.1 Minutes: blending short and long memory

Stage 3a estimates a player's chance of playing from his matches weighted by recency, with one memory length.
Its sensitivity check showed the best length depends on how far ahead the prediction is: about 10 days for the
next gameweek, 60 days for five weeks out.

The reason shows in the data. Across 1,111 player-seasons on the tuning seasons, the correlation between "played
60 minutes or more" in two matches, after removing each player's own average, is:

| Matches apart | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Correlation | 0.41 | 0.27 | 0.17 | 0.10 | 0.06 |

If only the last match mattered (a first-order Markov chain), these would fall as 0.41, 0.17, 0.07. They fall
much more slowly after the first step, at about 0.6 per match. That is the pattern of a persistent underlying role
with one-off noise on top: a single rest or early substitution says little about next month. For that structure
the appropriate forecast is a recency-weighted estimate that fades towards the player's longer-run level.

So each minutes quantity (chance of 60 or more minutes, chance of a shorter appearance) is computed twice, with a
short memory (`m_S`) and a long memory (`m_L`), and blended per fixture:

`m(d) = v(d) * m_S + (1 - v(d)) * m_L`, with `v(d) = 0.5^(d / fade)`

where `d` is the number of days from the deadline to the fixture. `fade` is tuned; the table suggests roughly two
matches, about 14 days, as the starting value. The short memory stays at 10 days and the long memory is tuned over
30, 60 and 120 days. Known injury return dates remain a Stage 6 input.

### 5.2 Tuning target

Stage 3a tuned on next-gameweek accuracy, `J_0`. The optimiser now relies on all five horizons, so tuning uses

`J = sum_{h=0..4} omega_h * J_h`, with `omega_h` proportional to `0.85^h`

where `J_h` is the 3a objective (RMSE and rank correlation relative to rebuilt FPL form) at horizon `h`. The
weights are fixed in advance and are deliberately not the optimiser's tuned `gamma`: predictions are tuned first
and the optimiser's settings second, with no loop between them. The 2025/26 season, already used once as the 3a
holdout, joins 2022/23 to 2024/25 as tuning data. It is the only season with defensive contribution points.

### 5.3 Reported alongside

- Accuracy in gameweeks 1 to 5, where the model leans on last season's data.
- Results with gameweeks close to a double or blank excluded. The archive holds the final fixture list, so past
  doubles appear earlier than they were announced, which flatters a multi-week planner more than a single-week one.

## 6. Evaluation

### 6.1 Season replay

For each past season a simulated manager starts gameweek 1 with a squad built from scratch for £100m, then every
week: predictions are made from data before the deadline only; the optimiser is solved; week 1's transfers are
applied; real points are scored; squad, bank, free transfers and purchase prices are carried forward. Buy prices
are the archive's price for that gameweek and selling prices follow FPL's rule (half of any rise, rounded down).
Chips are not played in the replay.

Scoring follows FPL: a starter who does not play is replaced by the first eligible bench player who did, keeping
a valid formation, a goalkeeper only by the bench goalkeeper; if the captain does not play, the vice-captain's
points are doubled instead. Without this the bench would be worth nothing and its weight could not be tuned.

### 6.2 Policies compared

- Single-week optimiser as it stands today.
- Single-week optimiser fed discounted five-week totals.
- Multi-period model at horizons of 3, 4 and 5 weeks.
- No transfers after gameweek 1, as a floor.

All policies use the same predictions, so the comparison isolates the planning method.

### 6.3 Settings and validation

Tuned by coordinate descent on total real points: `gamma`, horizon, bench weight and the scale of the leftover
transfer table. The replay also reports what hits actually returned; a margin `delta` is tuned only if they lost
points.

All four seasons are tuning data and the true test, the live 2026/27 season, takes months. For an unbiased
estimate from history, validation is leave-one-season-out: each season in turn is scored with the settings that
were best on the other three. Every candidate setting is replayed once on every season and each fold then picks
from those results, so the folds cost no extra solves. The reported figure is the total over the four held-out
seasons, with a gameweek-block bootstrap interval on the difference from the single-week policy.

Limits, stated in advance: about 150 gameweeks, each depending on the squad carried from the last, so intervals
will be wide and a real gain of a point or two per week may not be separable from noise. The prediction
parameters are tuned on all four seasons, which makes absolute totals optimistic but affects every policy alike.

### 6.4 Forward test

Each live run records the model's own predictions before the deadline, as it already records FPL's `ep_next`.
During 2026/27 these give an untouched comparison of the model against FPL's real forecast, and the weekly
recommendations give a record of what the multi-week plan advised.

## 7. Live use

`python -m optimisation.cli --entry ID` prints this week's transfers, XI, bench and captain as now, followed by
one line per later week: planned transfers, captain and expected points, marked as provisional. New options:
`--horizon`, `--discount`, and `--chip NAME:GAMEWEEK`. Tuned settings ship in the repository and can be overridden
by a local re-tune, in the same way as the prediction parameters.

## 8. Testing

- Stage 2's known-answer and brute-force tests, unchanged, at `H = 1`.
- New known-answer cases: a transfer is kept for a week where it gains more; a player is bought a week before a
  double gameweek; a hit is taken only when the gain over the horizon exceeds its cost; selling price limits a
  later week's budget; each chip's effect; the squad after a Free Hit.
- Brute-force enumeration of every plan on a scaled-down game over two and three weeks, with random free
  transfers, discount and leftover values; the model's objective must match.
- The validator applied to every solve, in tests, in the replay and live.
- HiGHS against CBC on the same instances.
- Auto-substitution scoring against hand-worked gameweeks (captain absent, no valid substitute, goalkeeper).
- Replay: a season is reproduced exactly on a second run; no prediction uses data from after its deadline.

## 9. Build order

1. Multi-period model, validator, cross-checks, chips, timings. Checkpoint: solve time; trimming and gap decisions.
2. Minutes blend and re-tuned predictions. Checkpoint: accuracy by horizon, before and after.
3. Season replay, leftover-transfer measurement, tuning, leave-one-season-out. Checkpoint: multi-week against
   single-week on real points.
4. Command-line output, README, methodology pages.

## 10. Error handling and known limits

- An infeasible plan (for example a budget that cannot fund any legal squad) raises the existing clear error.
- A chip named for a week outside the horizon, or two chips in one week, is rejected before solving.
- Fewer than `H` gameweeks left in the season: the horizon shrinks to what remains.
- Prices are frozen over the horizon; exact for this week, approximate for planned weeks, refreshed at each
  re-solve.
- A player who leaves the league mid-season stays in a replayed squad scoring zero until sold at his last price.
- The plan is built on expected points and does not value flexibility; `gamma` is the stand-in.

## 11. Future extension: planning under uncertainty

The natural successor is a scenario-based model: sample several joint outcomes for the coming weeks, share the
first week's decisions across all scenarios, let later weeks differ by scenario, and maximise the average. It
values flexibility directly instead of through a discount. It needs outcome distributions that Stage 3a does not
yet produce and multiplies the model's size by the number of scenarios. The only provision made for it now is
that predictions enter the optimiser as a plain table of player by week, to which a scenario index can be added.
