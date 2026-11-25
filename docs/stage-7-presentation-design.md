# Stage 7: Presentation (design)

Status: agreed with the user 2026-11-20 (brainstorm 2026-11-19/20, mockups in `.superpowers/brainstorm/504-1795124564/content/`, local). Built in sub-stages 7a-7f, each with its own local implementation plan in `docs/plans/` and its own PR.

## 1. Intent
Put the coach in an app: open it before a deadline, see your team and its predicted points, see what the coach would do, try your own swaps with instant feedback, read the report, ask follow-up questions, and check whether the model deserves trust. It must work on a phone as well as a laptop, look and feel like a polished product, and be robust and secure enough to open to other managers later without a rewrite.

It is a portfolio showcase. The owner shows it off personally; there is no public demo mode.

Success:
- The owner uses it instead of the CLI before each deadline in 2026/27.
- It meets the quality bar in section 9 on a small iPhone, a large Android phone and a laptop.
- Every number on screen comes from the pipeline (predictions, optimiser, coach run, monitoring); the app computes nothing new except the swap preview (section 6), which is cross-checked against the Python rules.

## 2. Goals, non-goals, anti-goals
- Goals: home screen with team and predictions; coach plan view; interactive swaps; report; in-app what-if chat; Model tab (trust + how it works); private developer analytics; deployed securely with logins, rate limits and a spend cap; scheduled pre-deadline coach runs.
- Non-goals (Stage 7): App Store / Play Store builds (later, via Capacitor); push notifications; public sign-ups and multi-user onboarding; light mode; a public demo mode; changing predictions, the optimiser or the coach's logic.
- Anti-goals (would be a failure even if it works): anything that looks broken or janky (raw errors, layout jumps, spinners with no progress, stutter); an API key, FPL data from the archive or another user's data reachable from the browser; a points number on screen that contradicts the pipeline; "points" used for the discounted planning objective; a swap preview that allows an illegal squad; AI spend that can run away.

## 3. User decisions (2026-11-19/20)
1. First screen: your current team with its predicted points; from there the coach's plan, your own swaps and the report.
2. Must work as a web app and on a phone; a later phone-app wrapper must not need a rewrite.
3. Single user (the owner) now, built multi-user-ready: robust and secure even as a demo.
4. Look: dark, numbers first (the "analyst" feel) combined with a large pitch.
5. Home screen: large pitch with tabs on top, data panel below on phone and beside it on laptop.
6. Review pass adopted: floodlit-pitch palette instead of generic black + acid green, one typeface (Archivo), status by symbol and percentage as well as colour, one headline story, larger text, plain labels.
7. Pitch: flat with floodlights and mowing stripes (mockup "F2"); each player is a minutes ring.
8. Swaps: tap a player, pick a replacement, see the points change; the screen must feel interactive; suggest the best swap for that position.
9. Swap impact: this week's change first, with a caveat line for the next-five-weeks effect.
10. Model tab serves both the owner (can I trust it?) and portfolio viewers (how it works).
11. Stack A (section 7): React + Vite + TypeScript PWA on Cloudflare Pages, FastAPI on Google Cloud Run, Cloud Run Jobs, Supabase (Postgres + Auth), Cloud Storage.
12. Stage 7 is built in sub-stages, not one go (section 10).
13. Quality bar: presentable and smooth; a showcase project.
14. Everything sits behind the owner's login; no demo mode.
15. Once the app is live, the cloud bucket is the master copy of `data/`. Local CLI runs pull first, push results back and take the same job lock, so the forward-test logs never diverge (2026-11-20).
16. Hosting stays on free tiers (about £0 a month): the engine scales to zero and is woken when the owner logs in; a designed "warming up" state covers the wait. No always-warm instances (2026-11-20).
17. Spending: single, occasional user; no money spent "just because"; automation must be easy to control and turn off (section 8, spend controls) (2026-11-20).

## 4. Brand and visual system
Mood: a floodlit night match. One bold element (the pitch); everything around it quiet.

Tokens (from the approved mockups; builder may tune values for contrast, not change the scheme):

| Token | Value | Use |
|---|---|---|
| `turf` | `#0E1A15` | page background |
| `surface` | `#15241D` | panels, ring centres |
| `line` | `rgba(226,236,224,.22-.32)` | chalk lines, dividers |
| `light` | `#F3F5EC` | primary text, floodlight white, primary buttons |
| `muted` | `#8FA096` | secondary text |
| `fit` | `#5FD38D` | player fine, gains |
| `doubt` | `#F3B23E` | player doubtful (25-75%) |
| `out` | `#F2564B` | player out or 0%, losses |

- Colour encodes player status and gain/loss only, never decoration.
- Status always has a second cue: `✕ 0%` (out), `! 75%` (doubt), shown on the pitch and in lists.
- Type: Archivo (Google Fonts, variable width). Condensed (wdth ~62) heavy weight with tabular figures for numbers and the headline; normal width for reading text. Sentence case labels; no ALL-CAPS eyebrows, no middle-dot meta strings.
- Developer analytics screen uses a monospace face, to mark it as the dev view.
- Dark only in Stage 7.

The pitch (signature element):
- Flat, top-down, attack at the top; mowing stripes and two floodlight glows from the top edge (mockup F2; source `.superpowers/brainstorm/504-1795124564/content/pitch-angles.html`, variant `mown`).
- Each starter is a disc: number = predicted points this GW (condensed, large); ring arc = probability of 60+ minutes (`p60` from the minutes model), coloured by status; name chip below; captain wears a "C" armband. Bench shown as a quieter row under the pitch.
- Key line under the pitch: "Ring = chance of playing 60+ minutes. Number = predicted points."
- Drawn as SVG (pitch) + HTML (players), responsive. Ring fills animate once on load and when a value changes (respecting reduced motion).

## 5. Screens
Bottom navigation (phone) / side or top navigation (laptop): **Team, Report, Ask the coach, Model**. Developer analytics is a separate route visible only to the owner's account.

**Team (home)**
- Header: team name; "Gameweek N deadline Sat 11:00 (38h)".
- Headline: `54.0 → 57.9`, captioned "Predicted points for your team now, and with the coach's plan". This week only; the five-week planning objective is never labelled points on this screen.
- Tabs: **Your team / Coach's plan / Try a swap**. Coach's plan shows the plan's GW1 squad with new signings marked; transfers count and hits shown.
- The coach's call: one sentence plus a reason and a button (e.g. "Play your Wildcard this week" / "See the Wildcard team").
- Injuries and doubts list (FPL chance and news); chips worth playing soon (from the chip scan, gains vs no chip).
- Phone: one scrolling column in that order. Laptop: pitch left, data panel right.
- Run status: when the latest coach run is older than the latest data or a deadline is near, show it and offer "Run the coach now" with live step progress (data, predictions, news, solve, chip scan, write-up).

**Try a swap** (section 6).

**Report**: the latest coach report rendered from its sections (Verdict, Why, Risks and news, Chip outlook, Alternative, Sources) with readable typography; the chip table as a real table; history of past reports by gameweek.

**Ask the coach**: the existing what-if chat (Stage 6 tools, gap 0.001 as today) as a chat screen; answers stream; each answer states exact vs 0.1% gap as today; post-check warnings shown discreetly. Uses the session of the latest coach run.

**Model**
- "Can I trust it?": season-to-date forward-test numbers from `evaluation/monitoring.py` (accuracy per model vs FPL `ep_next`, minutes calibration chart, planner points vs benchmarks), each with a one-sentence plain verdict; shows "not enough gameweeks yet" honestly when the sample is small.
- "How it works": a visual walk through the five layers (team ID -> data -> predictions -> optimiser -> AI review -> advice), each with a short plain explanation, its validated headline result, and a link to the methodology page in `docs/`.

**Developer analytics** (owner only, monospace): per run cost, tokens, cache hit rate, step timings, searches, flags and adjustments, tool calls, post-check warnings, report history (from `coach_log`, `prediction_log`, `data/reports/`); spend this month vs the cap.

## 6. Try a swap
Flow:
1. Tap a player on the pitch -> a sheet slides up: next-5 predicted points, p60, price (selling price), fixtures, FPL news.
2. Top of the sheet: the best swap for that position ("Best swap: Konsa, +3.9 this week"), then a ranked list. Every option respects budget (selling price + bank), squad composition and the 3-per-club cap; unaffordable or illegal options are greyed with the reason ("£0.4m short", "already 3 from Arsenal").
3. Pick one -> the pitch updates at once: ring and number change, the headline ticks to the new value, starting XI and captain re-pick if the swap changes who should start, a running tally shows the change including "−4 hit" beyond free transfers.
4. Stack several swaps, undo any, reset.
5. Impact line: this week's change first; under it the five-week effect, called out plainly when the two disagree ("but −2.1 over the next 5 weeks").
6. "Check against the coach's plan": runs a real solve with the swaps forced (existing `solve` force/ban, as the chat does) and returns a verdict with the multi-week knock-on. "Adopt this plan" (backlog item) is in scope for 7c: re-solve exactly with the swaps forced and make it the shown plan, labelled as the owner's choice.

Starting point (my call): swaps start from your current squad with your real free transfers and hit costs, whatever the coach's plan is. If a Wildcard or Free Hit is available, a toggle applies it to the swap set (no hits; Free Hit marked as reverting next week).

Five-week effect (my call): squad held fixed after the swaps (no further transfers), best XI and captain re-picked each week from the predictions, bench excluded, undiscounted, minus any hit; shown as "over the next 5 GWs (no further transfers)". It is deliberately simpler than the planner's objective, so "Check against the coach's plan" reports its own verdict, labelled as the optimiser's, and the screen says which number is which.

Adopt this plan (my call): stored as a separate "your plan" record beside the coach's run (snapshot and Postgres), shown as the owner's choice. It never overwrites the coach's headline, report or `plan_log` rows, so the forward test keeps scoring the coach's own decisions; it is logged as its own kind so the two can be compared later.

Instant preview (decided): the browser receives the week's prediction table (all players, horizon GWs, points and p60), the manager state (squad, selling prices, bank, free transfers) and the game rules, and computes the preview itself: squad legality, best XI and captain for the week, hit cost, and the five-week sum. Ranking "best swap for this position" = legal replacements ordered by this-week gain, five-week gain shown alongside.

Correctness (flagged per CLAUDE.md section 6): this duplicates rules that live in Python (squad composition, club cap, budget, formation, hit cost, XI/captain choice). The preview is a preview, never a plan; the Python side stays the authority. Required: golden test cases generated from the Python code (`optimisation/validate.py` legality, best-XI/captain selection, hit cost) checked by the TypeScript tests, so the two cannot drift silently; rules values come from `game_rules.json` via the API, never hard-coded in the front end.

## 7. Architecture (stack A)

Amendment (7c, see `docs/stage-7c-swaps-design.md`): the swap check solves inside the api, so there is no engine service for 7c; 7d decides whether chat needs an engine.
```
Phone / laptop browser (React PWA, Cloudflare Pages)
   | HTTPS + Supabase login token
   v
api (FastAPI, Cloud Run, light image: no torch, no solver calls)
   | reads: Supabase Postgres (app state) + run snapshots in Cloud Storage
   | starts: engine requests and Cloud Run Jobs
   v
engine (Cloud Run service, full pipeline image)      jobs (Cloud Run Jobs, same full image)
   chat turns, "check against plan" solves              coach run, retrain, data refresh
   \___________________ Cloud Storage bucket (private) _____________________/
        data/processed Parquet, data/models, data/reports, run snapshots
```
Decisions (mine, approved as part of this document):
- **Layers stay separate.** Presentation is a new `presentation/` area: `presentation/api/` (FastAPI) and `presentation/web/` (React). No prediction, optimisation or reasoning logic moves into it; it calls the existing packages.
- **Structured run snapshot (new output of the coach run).** Today a run produces Markdown plus logs; the app needs structure. `orchestration.coach.run` additionally writes `data/reports/<entry>_gw<N>_<stamp>.json`: squad and selling prices, bank, FTs, XI/bench/captain/vice now and in the headline plan per GW, transfers, prediction table (points per GW, p60) for all players, flags and adjustments, chip scan rows, report sections, notes. The report Markdown stays. The snapshot is the API's main read model and the input for the swap preview.
- **Two Cloud Run services + jobs from one codebase.** `api` is light and fast to start. `engine` loads the pipeline for chat (its tools solve) and swap checks; it rebuilds the chat `Session` from the latest snapshot and the stored data. Heavy, long work (coach run about 3 min, retrain about 16 min) runs as Cloud Run Jobs. `chat_loop` gains a request/response turn function for the API; the CLI loop stays.
- **Files in a bucket.** Jobs and engine sync the needed `data/` paths from the private bucket to local disk, run the unchanged file-based pipeline, and write results back. One writer at a time: a lock (Postgres row) serialises jobs. Running locally keeps working exactly as today (OneDrive `data/`).
- **App database (flagged new storage decision).** Supabase Postgres holds app state only: users, the user's FPL entry ID, runs (status, step progress, snapshot path, cost), saved swap sets, chat conversations, spend ledger. Parquet stays the pipeline's source of truth; the archive data never enters Postgres or the browser. Row-level security on every table.
- **Scheduled runs.** Cloud Scheduler triggers a daily check job (no AI cost). It starts at most one automatic coach run per gameweek, about 24 h before the deadline, only if automatic runs are switched on and no run exists for that GW yet; it also starts the retrain when due (compute only, no AI). The owner can start a run from the app at any time.
- **FPL reachability first (my call).** FPL sometimes refuses data-centre IPs. 7b starts with a check that a Cloud Run job can reach every FPL endpoint used; if not, the fallback is ingestion on the owner's PC uploading to the bucket, with the rest unchanged.
- **Local first.** Docker Compose runs web, api, engine and a local Postgres (Supabase CLI or plain Postgres) on the owner's PC; 7a needs no cloud account.

## 8. Security
- Auth: Supabase Auth; sign-ups disabled; the owner's account only (allowlist); every API request verifies the Supabase JWT; each user is bound to their own entry ID(s).
- Secrets (`ANTHROPIC_API_KEY`, `FOOTBALL_DATA_API_KEY`, Supabase service key) in Google Secret Manager for cloud, `.env` locally (existing `ingestion/env.py`); never in the browser bundle or the repo.
- Spend cap: monthly Anthropic budget covering every AI call made by the deployed app (coach runs, scheduled or manual; chat turns; grader if run in the cloud). Chat turns gain cost logging (today only the coach run logs `cost_usd`). Checked before starting a run and before each chat turn, never mid-run (a half-finished run wastes what it spent); hitting it blocks AI calls with a clear message and the rest of the app keeps working.
- Spend controls (decision 17), all visible on one Settings screen:
  - Monthly AI cap, default £5 (a coach run is about 9p), editable by the owner; spend so far shown.
  - "Automatic runs" switch (one coach run per gameweek); default on (my call: about £3.50 a season); off means nothing spends AI money unless the owner presses a button.
  - "AI off" switch: blocks every AI call at once (runs, chat); the app still shows the latest results and swaps still work.
  - Backstops outside the app, set up in 7b and documented: a monthly spend limit on the Anthropic API account (provider-level, so even a bug in the app cannot exceed it); a Google Cloud budget alert by email at £1; Supabase on the free plan with no card. A one-command "pause everything" script (pause Scheduler, block AI) in the runbook.
- Rate limits per user on all endpoints; stricter on chat, runs and swap checks.
- Bucket private; services use their own least-privilege service accounts; CORS limited to the app's origin; HTTPS only; security headers (CSP) on the front end.
- Report and chat Markdown rendered with a sanitising renderer (no raw HTML), since LLM text and web-search content reach the screen.

## 9. Quality bar (definition of done for every sub-stage)
- Home screen shows the team within about 1.5 s on a mid-range phone over 4G (cached snapshot); Lighthouse performance and accessibility 90+.
- Swap preview and ring animations smooth (no visible stutter); animation reduced when the OS asks.
- Every screen has designed loading, empty and error states that say what happened and what to do; no raw errors; no layout shift as data arrives; long runs show real step progress.
- One token set used everywhere; WCAG AA contrast on the dark palette; visible keyboard focus; text at least 11 px on phone.
- Checked on a small iPhone, a large Android phone and a laptop before a sub-stage is called done.
- Existing offline test suite stays green; new Python code tested with pytest, front end with unit tests (Vitest) plus a few end-to-end checks (Playwright) of the main flows.

## 10. Sub-stages
Each ends runnable, with its own plan, review and PR.
- **7a Local app skeleton.** Run snapshot JSON from the coach run; FastAPI read endpoints; React PWA with the design tokens, Team screen (pitch with rings, headline, tabs Your team / Coach's plan read-only, coach's call, issues, chips), Docker Compose. Owner-only login can be stubbed locally.
- **7b Deploy and security.** Supabase (Auth + Postgres + RLS), Cloud Run api/engine/jobs, bucket sync and job lock, Secret Manager, Cloudflare Pages, rate limits, Settings screen with the spend controls (cap, automatic runs, AI off) and the provider-level backstops, Scheduler, run-now with progress. App live behind the owner's login.
- **7c Try a swap.** Player sheet, instant preview, best swap per position, five-week caveat, stacking/undo, check against the coach's plan, adopt this plan; golden-case cross-checks.
- **7d Report and Ask the coach.** Report screen and history; chat turn API in the engine, streaming chat screen.
- **7e Model tab.** Monitoring-driven trust section; how-it-works walkthrough.
- **7f Developer analytics.** Owner-only dev screen.

Later (not Stage 7): Capacitor store builds, push alerts before deadlines, public sign-ups with per-user quotas, light mode.

## 11. Left to the builder
- Component structure, state management, routing and styling approach in React (keep it simple; no heavy UI kit that fights the design).
- Exact API endpoint shapes and the snapshot JSON schema (versioned).
- Chart library for the Model tab (follow the tokens).
- How the engine caches a rebuilt chat `Session` between turns.
- Exact token values within the scheme (contrast tuning), spacing scale, iconography.
- Choice of Postgres migration tool.
