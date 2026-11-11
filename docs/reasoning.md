# Reasoning layer

Stage 6 turned the pipeline into a coach. One command takes a team ID and returns a written recommendation:
transfers, captain, vice-captain and a chip call, with the reasons, the risks and the news behind them. A large
language model (Claude Sonnet 5.5, through the Anthropic API with tool use) reads the news, chooses whether to play
a chip and writes the report, but it never chooses players or states a number of its own: every squad comes from
the optimiser, every figure from a tool result, and code checks the text before it is shown.

Code: `orchestration/coach.py` (the pipeline), `reasoning/` (`client.py`, `news.py`, `flags.py`, `adjust.py`,
`chips.py`, `tools.py`, `report.py`, `chat.py`, `grader.py`, `prompts/`, `settings.json`),
`ingestion/congestion.py`, `ingestion/env.py`, `evaluation/monitoring.py`.

## How to use it

```
python -m orchestration.coach --entry <team ID> [--chat] [--no-search] [--horizon N] [--chip-gap X]
python -m evaluation.monitoring
python -m reasoning.grader [--all]
```

`--chat` opens a follow-up what-if chat after the report; `--no-search` skips web search (FPL's own news fields
only); `--horizon` overrides the planning horizon; `--chip-gap` sets a relative solver gap for the chip scan
(default 0, exact). The report is printed and saved to `data/reports/<entry>_gw<N>_<timestamp>.md`; the run's cost
is printed to the console and logged, not written into the report.

Keys are read from a local `.env` file in the project root, which git ignores: `ANTHROPIC_API_KEY` for Claude and
`FOOTBALL_DATA_API_KEY` for European and cup fixtures. A variable already set in the shell wins over the file. Keys
are never logged or stored anywhere else. Without Claude (no key, no network, an outage or a refusal) the coach
prints and saves the optimiser's plan with a note instead.

## One run

1. **Data.** The usual automatic refresh of the FPL data and the manager's state (squad, bank, free transfers,
   chips).
2. **Predictions.** The shipped ml-minutes predictions for the next five gameweeks, and an initial exact solve.
3. **News flags.** Claude reads a shortlist (the 15 owned players, the plan's buys, the best unowned players per
   position and the captain options), each with FPL's chance of playing and news text, each club's next fixture
   and any European or cup match within 4 days of it. It searches the web for press conferences, club injury
   updates and predicted lineups (at most 10 searches per run), and submits structured flags through a strict tool:
   status (fit, doubt, out, suspended, rotation risk), start probability, return gameweek, confidence, and a cited
   source with its URL, date and kind (official, media, predicted lineup). A flag is only for news beyond FPL's own
   fields. If a club's newest source is over 2 days old with the deadline within 48 hours, the report warns that
   its press conference may still be to come.
4. **Capped adjustments, in code.** `reasoning/adjust.py` decides what a flag may change; Claude cannot change a
   prediction directly. The rules:
   - A player FPL lists as ruled out or suspended cannot be raised.
   - Any player may be lowered on a cited source.
   - Raising is allowed only for a player FPL has flagged, and only when the source is newer than FPL's news.
   - A change is capped at 0.5 per gameweek, unless the flag is high confidence from an official source;
     predicted lineups count as medium confidence at most.
   - A return gameweek sets availability for later weeks (zero before it, the model's value from it); without
     one, the week-1 value fades over later weeks as FPL's chance of playing does.

   Injury, doubt and suspension flags change availability, which scales a player's chance of starting and of
   coming off the bench together. A rotation flag instead moves probability from starting to the bench in week 1,
   so a rested player still expects some minutes. Every flag, raw and applied value and cap is logged and shown in
   a table in the report.
5. **Headline solve.** The five-week optimiser is solved exactly on the adjusted predictions and re-checked by the
   independent rule validator.
6. **Chip scan.** The no-chip plan and every available chip in every horizon week (wildcard, free hit, bench boost,
   triple captain) are solved in parallel processes, and ranked by gain over no chip.
7. **Review and write-up.** Claude gets the squad state, the news flags and adjustments, the chip scan and any
   warnings up front, plus tools: `get_squad_state`, `get_predictions`, `get_news_flags`, `get_chip_scan`,
   `get_monitoring`, `solve` (a what-if with players banned from every week or forced into week one's squad, a
   chip or another horizon, at a 0.1% gap) and `choose_plan`. Squad and prediction rows carry FPL's chance of
   playing and news text, so a low projection is explained by its cause. It writes fixed sections: Verdict, Why,
   Risks and news, Chip outlook, an optional Alternative (a logged solve, cited by id), and Sources.
8. **Post-check, in code.** The players named in the Verdict must be exactly the headline's transfers, captain and
   vice-captain, and no one outside the week-1 squad or its sales; its Chip line must name the headline's chip and
   gameweek, or none. Every "N pts" figure in the text must match, to
   0.1, a figure from the headline, the chip scan or a logged solve. Mismatches are listed as warnings at the top
   of the report.
9. **Run log.** One row per run in `coach_log`: models, flags, adjustments, tool calls, token usage, cost and the
   report's path.

## The chip decision

The planner sees only its five-week horizon, so it cannot value a chip saved for later. Code therefore solves every
chip window and Claude makes the call: before writing it calls `choose_plan` with a scanned (chip, gameweek)
window, which makes that window's solved plan the headline, or with none to keep the no-chip plan. It is told each
chip's last playable gameweek (first-half chips expire after GW19), is reminded that an unused chip is worth
nothing, and must give a clear "play it now" or "hold it for GW N because ...". The headline is still always an
optimiser solve; when a chip is chosen, the no-chip plan is the natural alternative and its figures stay quotable.

## Fixture congestion

European and cup fixtures come from the football-data.org API (v4), mapped to FPL clubs and saved as a table.
The free tier was observed to include the Champions League; domestic cups and the other European competitions may be
missing, and the news step's web search covers that gap.

## Prompt caching and cost

Each request marks the system prompt as a cache breakpoint and adds an automatic breakpoint on the last block, so
each turn of a tool loop reads the previous turn's prefix from the cache. Measured on a live run: the second
tool-loop turn read 1,313 cached tokens and wrote 14. Requests use adaptive thinking with an effort level per role
(news medium, review high, chat medium, grader low) and the API's refusal fallback. Cost per run is computed from the
response usage and a price table in `reasoning/settings.json` (to be checked against the pricing page by hand).

Measured on three live runs for one team on 2026-11-11, all with Sonnet 5.5: **$0.081 to $0.095 per run**, about
3 minutes end to end. The chip scan, at 76 to 145 seconds, was the slowest step. Grading one report cost $0.018.

## What-if chat

With `--chat`, the report is followed by a terminal chat with the same tools apart from `choose_plan`: the
headline is fixed once written. What-ifs are solved at a 0.1% gap (`gap=0.001`) to keep answers quick, and every
answer says whether its figures are exact (the headline and exact chip-scan rows) or within 0.1% of optimal, citing
the solve id. Its "pts" figures are post-checked in the same way.

"What about player X?" works today as a what-if: the chat forces X into week one's squad with `solve` and reports
the best plan with him and its points against the headline. Planned next: a player lookup tool (minutes, xG/xA,
form, fixtures, price, FPL news) so the answer can show his stats, and an "adopt this plan" step that re-solves the
chosen what-if exactly and makes it the recommended plan, with the report re-rendered and post-checked.

## Monitoring report

`python -m evaluation.monitoring` scores the logged next-gameweek predictions of the current season against played
results: per model RMSE, rank correlation and mean error, minutes calibration (predicted against observed rate of
60+ minutes, by bucket), FPL's own `ep_next` as a benchmark, and the planner where available. It writes
`data/processed/monitoring.txt`, and the review can read a summary through `get_monitoring`. Its purpose is to spot
sustained drift before anything is re-tuned.

First 2026/27 numbers:

| Model | Rows | RMSE | Rank correlation |
|---|---|---|---|
| ml-minutes | 563 (1 gameweek) | 1.973 | 0.720 |
| component | 1,687 | 2.153 | 0.681 |
| FPL `ep_next` | 2,003 | 2.275 | 0.692 |

These are early, small samples, and the models are scored over different numbers of rows and gameweeks, so they
are not yet a comparison.

## Report grader

`python -m reasoning.grader` has Claude score the newest stored report not yet graded (`--all`: every ungraded
logged report) from 1 to 5 on five criteria: grounded numbers, sources cited, risks covered, clear moves and honest
uncertainty, with short notes. It sees the report and the run's logged flags, adjustments and tool calls, answers in
a fixed JSON schema (structured output, no tools), and the scores are appended to `grade_log`. It is meant to be
re-run when the prompts change, to show whether report quality holds.

Grading the third live report (gameweek 11, a wildcard recommendation): grounded numbers 2, sources cited 4, risks
covered 3, clear moves 3, honest uncertainty 3, for $0.018. The notes named the self-computed "7.4 pts" that the
post-check had already flagged, a miscount of injured players and little on rotation or congestion. It also
marked down a player named twice in the Transfers line, which was a false match by the post-check in that report's
warnings (since fixed): the grader takes the report's warnings at face value.

## Limits

- **No backtest of news effects.** There is no record of past news to replay, so whether the flags improve
  predictions can only be measured forward: the 2026/27 season, through the monitoring report.
- **The live runs so far had no news to find.** The 2026-11-11 runs were 10 days before the deadline, so web search
  found no press conferences or predicted lineups and the reports rest on FPL's own fields.
- **The post-check catches figures, not reasoning.** It flags a "pts" number Claude computed itself (as with a
  "7.4 pts" difference in the 2026-11-11 runs) and wrong players in the Verdict, but not a wrong count or claim in prose;
  the grader is the check for those, and it is itself a model's judgement.
- **One team, three runs.** The cost and time figures come from a single team in a single week.
- **No automated live test of the coach.** There is no `pytest -m live` coach run; every coach test uses a fake
  client. The live evidence is the C3 runs of 2026-11-11 against the real API.
