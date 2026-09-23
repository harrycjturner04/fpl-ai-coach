. # FPL AI Coach — Project Plan & Claude Code Instructions

This file is the source of truth for this project. Claude Code should read it at the start of any session in this repo, and **update the "Current Status" and "Changelog" sections at the end of any session that changes the plan, adds a component, or makes an architectural decision.** Everything else should only change when the human explicitly revises the plan.

## 1. Vision

An AI-assisted Fantasy Premier League coach. Input: a screenshot of the current squad plus available transfers/chips. Output: a data-grounded, explained recommendation for transfers, starting XI, and captaincy for the upcoming gameweek(s). The system combines a statistical/ML prediction layer, a constraint-based optimiser, and an LLM reasoning layer that incorporates qualitative context (injuries, news, rotation risk) and writes the final explanation.

This is a personal portfolio project, built incrementally, demonstrating both ML and optimisation skills alongside applied LLM engineering.

## 2. Architecture

Five layers, each with a single responsibility. Data flows top to bottom; only clean, structured output should cross a layer boundary.

1. **Ingestion** — pulls the raw state of the world: official FPL API (`bootstrap-static`, `fixtures`, `element-summary/{id}`, `entry/{id}`, `entry/{id}/history`, `entry/{id}/transfers`, `entry/{id}/event/{gw}/picks`), a vision-model parse of the user's squad screenshot into structured JSON, and injury/news scraping (Premier Injuries, Fantasy Football Scout, press-conference roundups) plus underlying stats (Understat/FBref for xG/xA).
2. **Feature engineering** — turns raw data into a versioned feature table: rolling form (3/5/10 GW), per-90 rates, fixture difficulty (FPL's FDR to start, own Elo-based rating later), home/away split, minutes reliability, set-piece duty, team strength.
3. **Prediction (ML)** — expected points per player for the next 1–5 gameweeks. Baseline: form weighted by fixture difficulty. Next: gradient-boosted trees (XGBoost/LightGBM) on the feature table. Minutes/start-probability should be modelled as its own signal, not folded silently into the points estimate.
4. **Optimisation (ILP)** — PuLP-based integer linear program. Maximises predicted points subject to budget, squad composition (2/5/5/3), max 3 per club, valid XI formation, captain doubling, and the -4-points-per-extra-transfer cost.
5. **Reasoning (LLM)** — reviews the ILP's output against qualitative context, converts unstructured injury/news text into structured risk flags the optimiser can consume, decides chip timing (wildcard/bench boost/triple captain/free hit), and writes the final natural-language recommendation. Should be able to call the optimiser as a tool to test scenarios rather than re-deriving squads itself.

An orchestration script ties these together per gameweek: screenshot → parsed team → fresh FPL data → features → predictions → ILP solve → LLM review and write-up → present.

## 3. Tech stack

**Decided:**
- Python for the data/ML/optimisation pipeline
- PuLP (CBC solver) for the ILP squad optimiser
- XGBoost or LightGBM for the prediction model (once past the baseline)
- Official FPL API for core data; no auth required
- Storage: flat files — immutable raw JSON snapshots per pull (`data/raw/<timestamp>/`) plus tidy Parquet tables (`data/processed/`). Chosen over SQLite and a vector DB: the data is tabular/exact-lookup, and timestamped raw snapshots give reproducible walk-forward re-runs. DuckDB can query the Parquet later without migration; a vector index may be added alongside (not instead) if Stage 6 needs semantic search over news.
- Stage 1 dependencies: `requests`, `pandas`, `pyarrow`; tests with `pytest`

**To be decided (revisit as the project develops):**
- Presentation layer: leaning Streamlit for a fast first dashboard, possible React/Next.js upgrade later if this becomes a polished portfolio piece
- Injury/news scraping approach and exact sources
- Which LLM/SDK integration pattern for the reasoning layer (direct Anthropic API + tool use is the working assumption)

## 4. Repository structure (proposed — adjust as it's built)

```
fpl-ai-coach/
  ingestion/        # FPL API client, screenshot parsing, news/injury scraping
  features/         # feature table construction, versioned
  prediction/        # baseline + ML models, training scripts, validation
  optimisation/      # ILP formulation and solver wrapper
  reasoning/          # LLM prompts, tool-use orchestration, report generation
  orchestration/     # end-to-end weekly pipeline script
  presentation/       # dashboard app
  data/               # raw pulls, feature tables, historical logs (gitignored as appropriate)
  tests/
  CLAUDE.md
```

## 5. Development roadmap

Build in this order — each stage should produce something runnable end-to-end before moving on, even if crude.

- [x] **Stage 1 — Ingestion skeleton.** Pull live FPL data (bootstrap-static, fixtures). Confirm data shapes and freshness.
- [ ] **Stage 2 — Optimiser on current stats.** Run the ILP against *current* season stats (no prediction yet) to prove constraint logic against obviously-checkable answers.
- [ ] **Stage 3 — Baseline prediction.** Form-weighted-by-fixture-difficulty model feeding the optimiser instead of raw current stats.
- [ ] **Stage 4 — Screenshot parsing.** Vision-model call turning a squad screenshot into structured JSON (team, bank, free transfers, chips available). Wire this in as the real input to the pipeline.
- [ ] **Stage 5 — ML prediction model.** Feature table + gradient-boosted model, walk-forward validated by gameweek (never randomly split — avoids leaking future form into the test set). Separate minutes/start-probability modelling.
- [ ] **Stage 6 — LLM reasoning layer.** Injury/news structuring, qualitative review of the ILP output, chip strategy, final written recommendation.
- [ ] **Stage 7 — Presentation.** Dashboard for viewing squad, predictions, and recommendations.

## 6. Instructions for Claude Code working in this repo

- Keep the five layers cleanly separated — don't let, e.g., prediction logic leak into the optimisation module. Each layer's module should be independently testable.
- Validate any prediction model with a walk-forward split by gameweek. Flag it clearly if a change risks leakage.
- The ILP constraints (budget, squad composition, per-club cap, valid formation, transfer cost) are the correctness backbone of this project — any change to them should be called out explicitly and tested against a known-correct squad before being trusted.
- Prefer incremental, runnable milestones over building multiple layers at once.
- When a genuine architectural decision gets made in a session (new library choice, a changed data source, a reworked layer boundary), update Section 3 and/or Section 5 above, and add a line to the Changelog below. Don't rewrite sections that haven't actually changed.
- Flag open questions rather than silently picking an approach when a decision materially affects later stages (e.g. database choice, LLM integration pattern).

## 7. Current status

*(Update this section as work progresses — it should always reflect where the project actually is, not the target state.)*

**Stage 1 complete (2026-09-23).** `ingestion/` package:
- `fpl_client.py` — `FPLClient` (retry/backoff, polite rate limit) for bootstrap-static, fixtures, element-summary, entry, entry picks.
- `storage.py` — raw JSON snapshots + Parquet read/write.
- `transform.py` — pure raw→DataFrame functions (players, teams, positions, gameweeks, fixtures, player_history, entry_picks); prices in £m.
- `cli.py` — `python -m ingestion.cli [--history] [--entry ID]`, prints a shape/freshness summary.
- `manager_state.py` — reconstructs what the public API doesn't expose: free transfers for next GW, chip status (8 chip instances, one set per season half), purchase/selling prices, and the persistent (pre-Free-Hit) squad. Output: `data/processed/manager_state_<id>.json` + `entry_*` tables (overview, leagues, gameweeks, chips, transfers, past seasons, picks, squad). Game rules (budget, club cap, FT cap, sell-on fee) come from bootstrap into `game_rules.json` rather than being hardcoded.
- Free-transfer rule verified empirically (`scripts/validate_free_transfers.py`): +1/GW capped at 5; Wildcard/Free Hit keep banked FTs but add none. 0 mismatches across 2,885 real manager-GWs; the alternative (+1 on chip weeks) gave 42 mismatches.
- 28 offline pytest tests pass. Live run verified: 667 players, 20 teams, 38 GWs, 380 fixtures, 3216 player-history rows (GW1–5); current GW 5; manager state verified on public entry 1 (incl. Free Hit revert).
- Known gaps: transfers made for the upcoming GW aren't public until its deadline; late joiners' initial purchase prices need `--history` to be exact (flagged otherwise). The authenticated `my-team` endpoint would give exact values — deliberately not used (would require storing FPL login).

Not yet covered in ingestion (deferred to later iterations): Understat/FBref xG, injury/news scraping, screenshot parsing (Stage 4). Note FPL's own API now exposes xG/xA per player and per match, which may reduce the need for Understat.

Next: Stage 2 — PuLP ILP optimiser run on current stats.

## 8. Changelog

- **2026-09-16** — Initial plan drafted (architecture, tech stack, roadmap).
- **2026-09-23** — Stage 1 built: `ingestion/` package (FPL client, storage, transforms, CLI) + tests. Decided storage = raw JSON snapshots + Parquet flat files. Project put under git and pushed to GitHub (`fpl-ai-coach`, public).
- **2026-09-23** — Added manager-state ingestion (entry history, transfers, chips, derived free transfers and selling prices) from public endpoints only; authenticated `my-team` endpoint deferred. Free-transfer rule validated against real data.
