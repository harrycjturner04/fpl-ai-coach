# Stage 7c: Try a swap (design)

Agreed with the owner 2026-11-25. Extends section 6 of `docs/stage-7-presentation-design.md`; where the two differ, this document wins.

## 1. Intent
Before a deadline the owner wants to test their own transfer ideas against the coach: tap a player, pick a replacement, see at once what it does to this week's points, then ask the real optimiser whether the idea holds up over the planning horizon, and keep the chosen idea as "your plan" beside the coach's. Typical use: run the coach right before the deadline on the latest news, try swaps on that fresh run, adopt one, then make the transfers in the FPL app. The app never changes the real FPL team.

## 2. Goals, non-goals, anti-goals
- Goals: player sheet; instant browser preview (legality, best XI and captain, hits, five-week sum); best swap per position plus ranked list with reasons for illegal options; stacking, undo, reset; Wildcard/Free Hit toggle; "Check against the coach's plan" (real exact solve); "Adopt this plan"; stale-plan handling; golden-case cross-checks between Python and TypeScript; team name title larger.
- Non-goals: making FPL transfers; seeing transfers already made in the FPL app before the deadline (FPL does not publish them; run the coach before transferring); chat and report screens (7d); the report post-check backlog fixes.
- Anti-goals: a preview that allows an illegal squad; a preview number that silently disagrees with Python's rules; the owner's plan leaking into the coach's headline, report, `plan_log` or forward test; a check that looks frozen (bare spinner); the five-week preview number labelled as the optimiser's objective or vice versa.

## 3. Approaches (where the real solve runs)
- **A. In the api (chosen by the owner).** Add `pulp` and `highspy` (small, no torch) to the light api image; solve in a worker thread from the snapshot plus the mounted `data/`. No new service, no extra cold start. Cost: the "no solver calls in api" line of the Stage 7 spec is amended; a solve shares the single api instance's CPU.
- B. New `engine` Cloud Run service (full image), as the Stage 7 spec planned. Rejected: a second service and image plus a 30-60 s cold start on top of the solve.
- C. Cloud Run Job run per check. Rejected: about 2 min per check and polling.

Consequence for 7d (to settle there): chat's tools are solves plus Anthropic calls, so chat may also fit in the api if its `Session` can be rebuilt from the snapshot; the engine service may never be needed. 7c's solve code is written so 7d can reuse it.

## 4. Decisions
**Solve.** One exact solve (gap 0). Measured 2026-11-25 on entry 8298351, GW12 snapshot, one forced player, owner's PC: gap 0.001 16.4 s, exact 24.0 s, same objective (228.41). So "Check" runs the exact solve and "Adopt" stores that result instantly (no second solve). Cloud Run may be slower (estimate 30-50 s); the wait shows a timed progress state ("Checking against the coach's plan, about 30 s") sized from the measured time. If the deployed time is far above 60 s, revisit (more vCPU, or gap 0.001 for check with an exact adopt).

**Starting point.** Swaps start from the current squad (not the coach's plan) with the real free transfers, bank, selling prices and hit cost from the snapshot. Wildcard/Free Hit toggle shown only when that chip is available: no hits; Free Hit marked as reverting next week (five-week sum uses the reverted squad after week one).

**Instant preview (browser, `swap.ts`, pure and UI-free).** Inputs come from the snapshot only. Computes: squad legality (composition, club cap, budget = bank + selling prices of those sold - prices of those bought), hit count and cost beyond free transfers, best XI and captain for this week (enumerate the formations allowed by the position limits; exact for 15 players), and the five-week effect: squad held fixed, best XI and captain re-picked each week, bench excluded, undiscounted, minus hits, labelled "over the next 5 GWs (no further transfers)". Best swap for a position = legal replacements ordered by this-week gain, five-week gain alongside; illegal ones greyed with the reason ("£0.4m short", "already 3 from Arsenal").

**Snapshot schema 2.** Adds the squad rules (composition, XI min/max per position, starting XI size, club cap, hit cost, from `game_rules.json` and `positions`) the solve inputs (unrounded scores; today's `points` are rounded to 0.1), and each player's fixtures in the horizon (opponent, home/away, FDR) for the sheet. Schema-1 snapshots still render; the swap tab says "Run the coach to enable swaps".

**Check against the coach's plan (api).** `POST /api/entries/{entry}/plan-checks` with the swap set and chip toggle. Inputs come from the snapshot only (owner's decision via the builder check: never the live `data/`, so the check cannot disagree with the preview after a price change or refresh); schema 2 therefore carries everything a solve needs (unrounded scores, club and position, prices, chip state). Settings: the shipped plan settings (`plan_params.json`, with the mounted `data/processed` override).
- What it tests (owner's decision): **exactly your swaps this week**. The whole week-1 squad (current squad minus sold plus bought, 15 players) is passed as `forced`; nothing banned. The optimiser plans weeks 2-5 around it. Works unchanged for Free Hit (the forced squad is the Free Hit squad, which reverts) and Wildcard. Existing ILP input; no formulation change.
- Baseline (owner's decision): the check carries the coach's headline chips; the owner's toggle replaces only a week-1 chip. One solve, compared with the headline's stored objective and week-1 points. If the headline is not exact (snapshot `exact` false), the verdict says so.
- Verdict: this week's points and the planning objective with your swaps vs the coach's headline, labelled as the optimiser's, beside the browser's five-week number with the screen saying which is which. One check at a time (a second gets "already checking"); about 10 checks per hour per user; no AI cost, so no spend check. Solve runs in a worker thread so other requests keep answering.

**Adopt this plan.** Stores the check's result in a new Postgres table `user_plans` (RLS on, no anon/authenticated grants, as the other tables): entry, gameweek, swap set, chip, the solved plan (weeks, XI, captain, transfers, objective), the snapshot it came from, adopted_at. One active plan per entry and gameweek; adopting again replaces it. Never written to `plan_log`, the report or the coach's snapshot. A separate log row of kind "owner" is kept so owner and coach decisions can be compared later. The Coach's plan tab gains a "Coach's / Yours" switch once a plan is adopted.

**Stale plan (owner's choice: keep it).** A newer coach run never changes or deletes the owner's plan. If the plan's snapshot is not the latest, it is marked "based on an older run", its squad re-scored in the browser on the latest predictions, shown beside the coach's new plan, with a "Re-check on latest run" button (same swaps, new predictions; adopting the result replaces the old plan). The plan clears after its gameweek's deadline.

**Golden cases.** A Python script generates JSON fixtures from the real code (`optimisation/validate.py` legality, best XI and captain selection, hit cost, budget) over varied squads, including edge cases (club cap hit, exact budget, formation limits, blank weeks, chip toggles). Vitest runs `swap.ts` against them; a pytest checks the committed fixtures match a fresh generation so the two cannot drift silently. Rule values always come from the snapshot, never hard-coded in the front end.

**Small extra.** Team screen team name larger (backlog item).

## 5. Constraints
- Quality bar of Stage 7 spec section 9 (smooth preview, reduced motion, designed loading/empty/error states, keyboard focus, AA contrast, checked on phone and laptop, Lighthouse 90+).
- Existing ILP untouched (forced/banned already exist); `check_plan` validates every solve as today.
- Layers stay separate: the api calls `optimisation.planning`; no optimisation logic in `presentation/`, except the flagged browser preview.
- api image stays without torch; max 1 instance (existing in-memory limiters).

## 6. Left to the builder
Exact endpoint and `user_plans` shapes; sheet and list component structure; how the timed progress state is drawn; golden-case generator sampling; how the owner-log row is stored (Parquet beside `plan_log` vs Postgres); api CPU setting if the deployed solve is slow.
