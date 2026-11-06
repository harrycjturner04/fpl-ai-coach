# Stage 6: LLM reasoning layer (design)

Status: agreed with the user 2026-11-06. Implementation plan kept locally in `docs/plans/`.

## 1. Intent
Turn the pipeline (ingestion -> ml-minutes predictions -> multi-week ILP) into a coach: one command takes a team ID and returns a grounded, explained recommendation, then answers follow-up what-ifs by re-running the optimiser. The LLM adds what the numbers miss (injury/rotation news, chip context) and writes the explanation; it never invents a number.

Success = (a) every number in a report or chat answer traces to a tool result; (b) the forward test on 2026/27 shows whether news adjustments improve predictions; (c) report quality holds when prompts change (LLM grader).

## 2. Goals / non-goals / anti-goals
- Goals: report + follow-up chat; news -> structured flags -> bounded, logged availability changes; chip scan with LLM judgement; per-run cost reporting; weekly monitoring report.
- Non-goals: dashboard (Stage 7); season-end re-tune command and `--pending-transfer` (backlog); own scrapers; LLM-predicted points; fine-tuning; vector DB/RAG; multi-LLM debate; LLM captaincy override.
- Anti-goals: an LLM-stated number not from a tool; a silent change to predictions; an illegal squad; a headline recommendation that is not the optimiser's solve; API key or FPL login stored in the repo.

## 3. User decisions (2026-11-05)
1. Output: written report first, then follow-up what-if chat.
2. Own app calling the Anthropic API directly with tool use (Python `anthropic` SDK).
3. News power: LLM emits structured flags; code maps them to capped availability changes; logged and shown; optimiser re-solves.
4. Sources: FPL `news`/`chance_of_playing` plus Anthropic server-side web search (`web_search_20260209`) with citations; no own scrapers.
5. Chips: code scans no-chip + each available chip x each horizon week (real solves, parallel); LLM judges and recommends.
6. Cost: balanced; model per role configurable; search count capped; every run reports tokens, searches and cost so information-vs-cost can be evaluated later.
7. Evaluation: forward test on 2026/27 (raw vs adjusted predictions, minutes calibration, RMSE, planner points) + LLM grader of stored reports against a rubric.
8. Chat what-ifs solve at 0.1% gap; report's recommendation stays exact; each answer states which.
9. Scope extra: weekly monitoring report.
10. Adjustment rules (section 5) accepted as proposed.
11. Disagreement: headline is always the optimiser's solve; LLM may add one solved alternative with points cost and reason.
12. Rotation/resting: "rotation risk" flag; fixture congestion (European/cup) as a structured input; shortlist-driven search; stale-news warning before press conferences.
13. Flag maths: two mechanisms, availability vs start-to-bench shift (2026-11-06).
14. Congestion data: football-data.org API (2026-11-06).
15. Models: Sonnet 5.5 for every role (2026-11-06).

## 4. Pipeline (one run)
`python -m orchestration.coach --entry ID [--chat] [--no-search] [--horizon N]`
1. Refresh data (existing `ingestion/freshness.py` auto-refresh), manager state (`ingestion/manager_state.py`).
2. Raw predictions (`prediction.cli` path, shipped ml-minutes; logged to `prediction_log` as today).
3. Initial exact solve (`optimisation/model.py:solve_plan`, shipped `plan_params.json`) -> shortlist = owned 15 + transfer-in candidates (top-N by predicted points per position not owned, plus the plan's buys) + captain options.
4. News step: FPL news for shortlist players + fixture congestion + web search on shortlist clubs (press conferences, predicted lineups) -> structured flags (section 5).
5. Apply flags in code -> adjusted predictions (logged as model `<shipped>+news`).
6. Exact re-solve on adjusted predictions = headline recommendation; `validate.check_plan` runs on it.
7. Chip scan (parallel processes, exact or 0.1% gap: builder's call, state which) for each available chip x horizon week.
8. Review + write-up LLM call with tool access -> Markdown report (printed, saved `data/reports/<entry>_gw<N>_<timestamp>.md`).
9. Optional chat loop (`--chat`): terminal REPL; same tools; what-ifs at 0.1% gap.
10. Log run to `coach_log` (flags, raw vs adjusted availability, caps applied, tool calls, models, tokens, searches, cost, report path).

## 5. News flags and adjustment rules
Flag schema (per player): `player_id`, `status` in {fit, doubt, out, suspended, rotation_risk}, `start_prob_next_gw` [0,1], `return_gw` (nullable), `confidence` in {low, medium, high}, `source_url`, `source_date`, `source_kind` in {official (club/manager quote, FPL), media, predicted_lineup}, `summary` (one line). Flags only for new information beyond FPL fields and recent-minutes patterns (prompt instruction).

Rules enforced in code (`reasoning/adjust.py`, pure, unit-tested):
1. FPL ruled out (chance 0) or suspended is a hard floor: availability cannot be raised.
2. Lowering allowed for any player with a cited source.
3. Raising allowed only for FPL-flagged players and only when `source_date` > FPL `news_added`.
4. Cap: |change| <= 0.5 availability per GW unless `confidence == high` and `source_kind == official`. Predicted lineups are at most medium.
5. `return_gw` sets availability for later horizon weeks (0 before, model value from it) instead of the `flag_fade` decay.
6. Every flag, raw and adjusted value, and cap hit is logged and listed in the report.

Two mechanisms (user, 2026-11-06):
- Injury/suspension/out/doubt flags -> availability: replace the FPL `chance` term in `availability = 1 - (1 - chance) * flag_fade ** horizon` (`prediction/component_model.py:110-113`, `prediction/walkforward.py:713-717`), which scales `p60` and `psub` together. New optional per-player availability override input; without `return_gw` the week-1 value decays with `flag_fade`.
- Rotation flags -> start vs bench: move probability from `p60` to `psub` so the start chance matches the flag's `start_prob_next_gw` (fit players still come off the bench). Delivered through the existing per-fixture `minutes` override (`component_model.py:104-109`; check the ml-minutes path in `walkforward.py` takes the same route). Week 1 only unless the source covers later fixtures.
- The 0.5 cap (rule 4) applies to whichever probability is changed. Leakage: none (live only; backtests unaffected).

Staleness: report shows the newest source date per club and warns if a club's pre-match press conference is likely still to come (deadline within ~48 h and newest source older than ~2 days; builder's call on the exact rule).

## 6. Optimiser changes (flagged per CLAUDE.md section 6)
- `banned` players: drop from the pool (and if owned, force sale) for what-ifs.
- `forced` players: new constraint "in squad (and optionally XI) in week 1 / all weeks". ILP constraint change: tested against a known-correct squad and brute-force cross-check; `validate.check_plan` gains the same check independently.
- Gap setting: `solve_plan(..., gap=0.0)` passes `gapRel` to `pulp.HiGHS` (`optimisation/model.py:251`); default stays 0.

## 7. LLM tools (client tools, strict schemas)
`get_squad_state`, `get_predictions(players|position|top_n)`, `get_news_flags`, `solve(ban, force, chip, horizon, gap)`, `get_chip_scan`, `get_monitoring`, plus server `web_search` (news step and chat). Tools return compact JSON; the LLM never does arithmetic on points it cannot cite. Report numbers are cross-checked against the solve objects before printing (post-check in code; mismatch -> warning in report).

## 8. Models and cost
- Roles: news step (search + flags), review/write-up, chat, grader. Each configurable in `reasoning/settings.json`.
- Defaults (user): `claude-sonnet-5-5` for every role; adaptive thinking, effort set explicitly per role; `fallbacks: "default"` refusal fallback enabled (Claude API). Opus 5.5 remains a config switch for the review role if graded quality is lacking.
- Structured flags via strict tool or `output_config.format` (forced tool_choice is a 400 on these models; verify web search + structured output compatibility at build; fallback two-step: search call then extraction call).
- Prompt caching on the stable system prompt + tool list; `max_uses` cap on web search (default ~10, configurable).
- Cost per run computed from `usage` + search count with a price table in settings; printed and logged.
- Key: SDK default credential resolution (`ANTHROPIC_API_KEY` or `ant auth login` profile); never stored.

## 9. Fixture congestion input
Structured list of each shortlist club's non-PL fixtures within +/- 4 days of each horizon GW. Source (user): football-data.org API, new `ingestion/` module, key in env var `FOOTBALL_DATA_API_KEY` (never stored), saved as a Parquet table; team names mapped to FPL team codes. Verify free-tier competition coverage at build; gaps (e.g. domestic cups) filled by web search in the news step. New data source: CLAUDE.md section 3 updated. Tests use recorded JSON.

## 10. Monitoring report (`evaluation/monitoring.py`)
From `prediction_log` (now including `<shipped>+news`) and played results (`match_log`): season-to-date per model RMSE, rank correlation, minutes calibration (predicted vs observed 60+, by bucket), vs component and FPL `ep_next` (`ep_next_log`); planner points vs benchmarks where available. CLI `python -m evaluation.monitoring`; the coach reads its summary via `get_monitoring`.

## 11. Layout
- `reasoning/`: `client.py` (SDK wrapper, cost accounting, fallbacks), `news.py` (shortlist, search, flag extraction), `adjust.py` (rules), `tools.py` (tool schemas + handlers calling prediction/optimisation), `report.py` (write-up + number post-check), `chat.py` (REPL), `grader.py` (rubric scoring of stored reports), `prompts/`, `settings.json`.
- `orchestration/coach.py`: pipeline section 4.
- `prediction/`: availability override input only. `optimisation/`: banned/forced/gap. `evaluation/monitoring.py`. New dependency `anthropic`.

## 12. Testing
- Offline (default `pytest`): recorded/fake LLM client (no network); `adjust.py` rule table tests (each rule, caps, floors, return GW); flag schema validation; forced/banned ILP known-answer + brute-force + validator; gap plumbing; report number post-check catches a wrong number; tool handlers on fixture data; cost accounting from canned `usage`; monitoring metrics on a synthetic log.
- Golden scenarios with recorded responses: ruled-out star, "doubtful" wording, rotation quote before a CL game, DGW chip week; assert flags, adjustments and that the report cites them.
- Live (`pytest -m live`, opt-in, costs money): one real coach run on entry 8298351 with search capped.
- Grader: rubric (grounded numbers, sources cited, key risks mentioned, clear moves) over stored reports; run when prompts change; scores logged.
- Forward test: weekly via monitoring report.

## 12b. Calls made in design review (Claude's; open to revision)
- Flag without `return_gw`: week-1 value decays over later weeks with the existing `flag_fade` (same as FPL chance today).
- Ban/force is wholly new ILP work, not an extension.

## 13. Open questions
None outstanding. Builder check done; flag maths, congestion source and models decided by the user 2026-11-06.

## 14. Next steps after approval
1. Pull main, branch `stage-6`; commit this design as `docs/stage-6-reasoning-design.md`; update CLAUDE.md (PR #3 merged, Stage 6 design agreed, new sources/dependencies `anthropic` + football-data.org).
2. superpowers:writing-plans -> local plan `docs/plans/2026-11-06-stage-6-reasoning.md`; summarise in chat as a task list with checkpoints (proposed: C1 ILP ban/force + overrides tested; C2 news flags + adjust on golden scenarios; C3 first live coach report + cost; C4 chat, grader, monitoring).
3. Build only after the user approves the plan summary.

## Verification (end of build)
`pytest` all green; `pytest -m live` coach run on 8298351 produces a report whose numbers match the logged solves, flags cite sources, cost printed; `python -m evaluation.monitoring` runs on the current season's logs; grader scores the live report.
