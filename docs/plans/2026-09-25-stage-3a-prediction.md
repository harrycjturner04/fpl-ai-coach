# Stage 3a: Component Prediction Model and Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Stage 2 placeholder score with a component model of expected FPL points per player per fixture for the next five gameweeks, proven by a walk-forward backtest against FPL's own historical `xP`.

**Architecture:** A unified `match_log` (past seasons from the vaastav archive plus the current season from FPL's `event/{gw}/live`) feeds cutoff-aware feature functions (`features/`): Poisson team ratings fitted to xG, and shrunk per-player rates. `prediction/component_model.py` turns features, fixtures and injury flags into per-component expected points using per-season scoring rules. `prediction/backtest.py` replays past gameweeks with the same code path; `prediction/tune.py` fits parameters by coordinate descent. The optimiser's ILP is unchanged; only its input score changes.

**Tech Stack:** Python 3.12+, pandas, numpy, scipy (new), PuLP (existing), pytest.

**Spec:** `docs/stage-3a-prediction-design.md`

## Global Constraints

- Python `>=3.12`; new dependency `scipy>=1.11` (confirm a wheel installs on the local Python 3.14).
- No em dashes in any code, comment, doc or commit message.
- Every feature function takes a `cutoff` and reads only `match_log` rows with `kickoff < cutoff`.
- Tuning seasons: `2022-23`, `2023-24`, `2024-25`. `2025-26` is used only in Task 11, once.
- Archive data is never committed or republished (`data/` is excluded from git).
- Positions are always `GKP`, `DEF`, `MID`, `FWD` (archive `GK` is mapped; archive `AM` rows are dropped).
- Prices in the match log are in £m (archive `value` / 10).
- The optimiser's ILP constraints do not change in 3a.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A player who changed club mid-season: predictions must use his latest club (before the cutoff) and his rates must be normalised by the team context they were earned in. Pinned in Task 7.
2. Double and blank gameweeks in the fixtures ahead: a double sums two fixtures, a blank gives no rows (0 points). Pinned in Task 8.
3. A player with no history (new signing): falls back to position and price-band priors and is flagged `no_history`. Pinned in Task 8.
4. A promoted team at GW1 with no previous-season data: gets the promoted prior and a finite expected-goals value. Pinned in Task 6.
5. The current gameweek part-played when live data is pulled: only finished fixtures enter the match log. Pinned in Task 4.

## Notes on deviations from the spec (for the reviewer)

- **Build order:** the spec's steps 4 to 7 (add components one at a time with a checkpoint after each) are implemented as one component model with a `components` switch, followed by an **ablation checkpoint** (Task 9) that reports what each component adds by switching it off. Same evidence, less rework.
- **Promoted-team prior:** implemented as two tunable constants (`promoted_attack`, `promoted_defence`) rather than an average computed from past promoted teams; tuning picks the value the data supports.
- **FPL `xP` benchmark exists only for the next gameweek** (it is published per gameweek at its deadline). Horizons 1 to 4 are compared against the naive benchmark only.
- **GK goal value for 2025/26** is taken as 10 (the current rule); it cannot be verified from the archive because no goalkeeper scored that season.

---

## File Structure

| File | Responsibility |
|---|---|
| `optimisation/cli.py` (modify) | `data_dir` parameter; injectable refresh; `--scorer component` (Task 12) |
| `prediction/scoring.py` (new) | FPL points per component per position per season; points reconstruction from stats |
| `ingestion/archive.py` (new) | Download and convert archive seasons to match-log rows |
| `ingestion/fpl_client.py` (modify) | `event_live(gw)` |
| `ingestion/transform.py` (modify) | `MATCH_COLUMNS`, `season_label`, `live_match_rows` |
| `ingestion/storage.py` (modify) | `load_table_or_none` |
| `ingestion/cli.py` (modify) | Maintain current-season rows cache and write `match_log.parquet` |
| `features/__init__.py` (new) | Package marker |
| `features/team_ratings.py` (new) | `team_matches`, `TeamParams`, `TeamRatings`, `fit_team_ratings` |
| `features/player_rates.py` (new) | `PlayerParams`, `price_band`, `player_features` |
| `prediction/poisson.py` (new) | `expected_floor_div` |
| `prediction/component_model.py` (new) | `ModelParams`, `fixture_components`, `predict`, `per_gameweek` |
| `prediction/backtest.py` (new) | `Snapshot`, benchmarks, metrics, decision value, `run_backtest`, CLI |
| `prediction/tune.py` (new) | Coordinate-descent tuning, `model_params.json` |
| `prediction/cli.py` (new) | Live five-week predictions table and `predictions.parquet` |
| `pyproject.toml` (modify) | `scipy`, `features` package |

---

### Task 1: Offline test for the optimiser CLI

**Files:**
- Modify: `optimisation/cli.py`
- Test: `tests/test_optimisation_cli.py`

**Interfaces:**
- Consumes: `optimisation.model.solve`, `tests.optimiser_helpers.full_pool`, `pool`, `FULL_RULES`, `ingestion.storage`.
- Produces: `run(entry_id=None, ep_weight=0.7, bench_weight=0.1, max_transfers=None, budget=None, ft_value=1.5, data_dir=storage.DATA_DIR) -> tuple[Solution, dict]`; `ensure_fresh_data(entry_id, data_dir=storage.DATA_DIR, refresh=None, now=None) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_optimisation_cli.py
from datetime import datetime, timedelta, timezone

import pandas as pd

from ingestion import storage
from optimisation import cli
from optimisation.model import solve
from tests.optimiser_helpers import FULL_RULES, full_pool, pool

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)


def _write_data(tmp_path):
    players, _ = pool(full_pool())
    players = players.assign(team_short="T", ep_next=2.0, form=2.0,
                             chance_of_playing_next_round=None, can_select=True)
    storage.save_table(players, "players", tmp_path)
    storage.save_table(pd.DataFrame({
        "position": ["GKP", "DEF", "MID", "FWD"], "squad_select": [2, 5, 5, 3],
        "squad_min_play": [1, 3, 2, 1], "squad_max_play": [1, 5, 5, 3]}), "positions", tmp_path)
    storage.save_json({"starting_xi": 11, "max_per_club": 3, "hit_cost": 4,
                       "max_free_transfers": 5, "total_budget": 100.0}, "game_rules", tmp_path)
    return players


def _fresh_metadata(tmp_path, entries=None):
    storage.save_json({"pulled_at": NOW.isoformat(), "next_deadline": (NOW + timedelta(days=5)).isoformat(),
                       "entries": entries or {}}, "metadata", tmp_path)


def test_from_scratch_run_uses_data_dir(tmp_path):
    _write_data(tmp_path)
    sol, context = cli.run(data_dir=tmp_path)
    assert len(sol.squad) == 15 and context["mode"] == "from scratch"
    assert "Starting XI" in cli.format_solution(sol, context)


def test_transfer_run_reads_manager_state(tmp_path):
    players = _write_data(tmp_path)
    scores = pd.Series(1.0, index=players["id"])
    owned = solve(players, scores, FULL_RULES, budget=100.0).squad
    storage.save_json({"team_name": "Test FC", "bank": 0.0, "free_transfers": 1,
                       "squad": [{"player_id": int(i), "selling_price": 4.5} for i in owned]},
                      "manager_state_42", tmp_path)
    sol, context = cli.run(entry_id=42, data_dir=tmp_path)
    assert context["mode"] == "transfers for Test FC"
    assert sol.free_transfers_next is not None


def test_stale_data_triggers_refresh(tmp_path):
    calls = []
    cli.ensure_fresh_data(None, data_dir=tmp_path, now=NOW,
                          refresh=lambda: (calls.append(1), _fresh_metadata(tmp_path)))
    assert calls == [1]  # no metadata yet -> refresh


def test_fresh_data_does_not_refresh(tmp_path):
    _fresh_metadata(tmp_path)
    calls = []
    status = cli.ensure_fresh_data(None, data_dir=tmp_path, now=NOW, refresh=lambda: calls.append(1))
    assert calls == [] and status.startswith("Data pulled 0 min ago")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_optimisation_cli.py -v`
Expected: FAIL (`run()` got an unexpected keyword argument `data_dir`).

- [ ] **Step 3: Implement**

In `optimisation/cli.py`, replace `ensure_fresh_data` and the start of `run`:

```python
def ensure_fresh_data(entry_id: int | None, data_dir: Path = storage.DATA_DIR,
                      refresh=None, now: datetime | None = None) -> str:
    """Re-pull the data if it's too old to optimise on; returns a one-line status."""
    now = now or datetime.now(timezone.utc)
    reasons = freshness.stale_reasons(storage.load_json_or_none("metadata", data_dir), now, entry_id)
    if reasons:
        print(f"Refreshing data ({'; '.join(reasons)})...")
        (refresh or (lambda: ingestion_cli.run(entry_id=entry_id, data_dir=data_dir)))()
    pulled_at = datetime.fromisoformat(storage.load_json("metadata", data_dir)["pulled_at"])
    minutes = (now - pulled_at).total_seconds() / 60
    return f"Data pulled {minutes:.0f} min ago ({pulled_at:%a %d %b %H:%M} UTC)"


def run(entry_id: int | None = None, ep_weight: float = 0.7, bench_weight: float = 0.1,
        max_transfers: int | None = None, budget: float | None = None,
        ft_value: float = 1.5, data_dir: Path = storage.DATA_DIR) -> tuple[Solution, dict]:
    players = storage.load_table("players", data_dir)
    game_rules = storage.load_json("game_rules", data_dir)
    rules = SquadRules.from_data(game_rules, storage.load_table("positions", data_dir))
```

and change `storage.load_json(f"manager_state_{entry_id}")` inside `run` to `storage.load_json(f"manager_state_{entry_id}", data_dir)`. Add `from pathlib import Path` to the imports.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q`
Expected: all pass (119 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add optimisation/cli.py tests/test_optimisation_cli.py
git commit -m "Optimiser CLI: data_dir and injectable refresh, with offline tests" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Scoring rules per season

**Files:**
- Create: `prediction/scoring.py`
- Test: `tests/test_scoring_rules.py`

**Interfaces:**
- Produces: `ScoringRules` (frozen dataclass: `goal`, `clean_sheet`, `concede_per_two` dicts by position; `assist`, `save_per_three`, `yellow`, `red`, `own_goal`, `penalty_saved`, `penalty_missed`, `defensive_contribution` ints; `dc_threshold` dict), `rules_for_season(season: str) -> ScoringRules`, `points_from_stats(rows: pd.DataFrame, rules: ScoringRules) -> pd.Series`, `DC_FIRST_SEASON = "2025-26"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring_rules.py
import pandas as pd

from prediction.scoring import points_from_stats, rules_for_season


def _row(**kw):
    base = dict(position="MID", minutes=90, goals=0, assists=0, clean_sheets=0, goals_conceded=0,
                own_goals=0, penalties_saved=0, penalties_missed=0, saves=0, bonus=0,
                defensive_contribution=0, yellow_cards=0, red_cards=0)
    base.update(kw)
    return base


def test_rules_change_in_2025_26():
    old, new = rules_for_season("2024-25"), rules_for_season("2025-26")
    assert old.goal["GKP"] == 6 and new.goal["GKP"] == 10
    assert old.defensive_contribution == 0 and new.defensive_contribution == 2


def test_points_from_stats_known_rows():
    rows = pd.DataFrame([
        _row(position="DEF", goals=1, clean_sheets=1, bonus=3),                 # 2 + 6 + 4 + 3
        _row(position="GKP", saves=7, goals_conceded=3, yellow_cards=1),        # 2 + 2 - 1 - 1
        _row(position="FWD", minutes=20, assists=1),                            # 1 + 3
        _row(position="MID", minutes=0),                                        # 0
        _row(position="DEF", defensive_contribution=10),                        # 2 + 2 (2025-26 only)
    ])
    assert points_from_stats(rows, rules_for_season("2025-26")).tolist() == [15, 2, 4, 0, 4]
    assert points_from_stats(rows, rules_for_season("2024-25")).tolist() == [15, 2, 4, 0, 2]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_scoring_rules.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# prediction/scoring.py
"""FPL points per component, per position, per season.

Verified 2026-09-25 by rebuilding every archive row's points from its stats:
100% exact for 2022/23 to 2025/26 (2024/25 excluding Assistant Manager rows).
The GK goal value for 2025/26 (10) is the current rule; no goalkeeper scored
that season, so it could not be checked against data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DC_FIRST_SEASON = "2025-26"


@dataclass(frozen=True)
class ScoringRules:
    goal: dict[str, int]
    clean_sheet: dict[str, int]
    concede_per_two: dict[str, int]
    defensive_contribution: int
    assist: int = 3
    save_per_three: int = 1
    yellow: int = -1
    red: int = -3
    own_goal: int = -2
    penalty_saved: int = 5
    penalty_missed: int = -2
    dc_threshold: dict[str, float] = field(
        default_factory=lambda: {"GKP": np.inf, "DEF": 10, "MID": 12, "FWD": 12})


def rules_for_season(season: str) -> ScoringRules:
    new = season >= DC_FIRST_SEASON
    return ScoringRules(
        goal={"GKP": 10 if new else 6, "DEF": 6, "MID": 5, "FWD": 4},
        clean_sheet={"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0},
        concede_per_two={"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0},
        defensive_contribution=2 if new else 0,
    )


def points_from_stats(rows: pd.DataFrame, rules: ScoringRules) -> pd.Series:
    """Actual FPL points from a match's stats (used to validate the rules)."""
    pos = rows["position"]
    appearance = np.where(rows["minutes"] >= 60, 2, np.where(rows["minutes"] > 0, 1, 0))
    dc_hit = rows["defensive_contribution"] >= pos.map(rules.dc_threshold)
    points = (
        appearance
        + pos.map(rules.goal) * rows["goals"]
        + rules.assist * rows["assists"]
        + pos.map(rules.clean_sheet) * rows["clean_sheets"]
        + pos.map(rules.concede_per_two) * (rows["goals_conceded"] // 2)
        + (pos == "GKP") * rules.save_per_three * (rows["saves"] // 3)
        + rules.penalty_saved * rows["penalties_saved"]
        + rules.penalty_missed * rows["penalties_missed"]
        + rules.yellow * rows["yellow_cards"]
        + rules.red * rows["red_cards"]
        + rules.own_goal * rows["own_goals"]
        + rows["bonus"]
        + rules.defensive_contribution * dc_hit
    )
    return points.astype(int)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_scoring_rules.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prediction/scoring.py tests/test_scoring_rules.py
git commit -m "Prediction: per-season FPL scoring rules" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Archive ingestion

**Files:**
- Create: `ingestion/archive.py`
- Modify: `ingestion/transform.py` (add `MATCH_COLUMNS`)
- Test: `tests/test_archive.py`, and a `live`-marked test in `tests/test_live_api.py`

**Interfaces:**
- Consumes: `prediction.scoring` (live test only), `ingestion.storage`.
- Produces: `transform.MATCH_COLUMNS: list[str]`; `archive.ARCHIVE_SEASONS`; `archive.season_match_rows(merged, players_raw, teams, season) -> pd.DataFrame[MATCH_COLUMNS]`; `archive.fetch_season(season, session=None) -> dict[str, pd.DataFrame]` (keys `merged`, `players_raw`, `teams`); `archive.load_archive(seasons=ARCHIVE_SEASONS, fetch=fetch_season) -> pd.DataFrame`; CLI `python -m ingestion.archive` writing `data/processed/archive_match_log.parquet`.

- [ ] **Step 1: Add `MATCH_COLUMNS` to `ingestion/transform.py`** (below `HIT_COST`)

```python
# One row per player per fixture, for past seasons (archive) and this season (event/live).
MATCH_COLUMNS = [
    "season", "gameweek", "fixture_id", "kickoff", "player_code", "team_code", "opponent_code",
    "was_home", "position", "minutes", "starts", "goals", "assists", "xg", "xa", "xgc",
    "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed", "saves",
    "bonus", "defensive_contribution", "yellow_cards", "red_cards", "points", "price", "xp",
]
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_archive.py
import pandas as pd
import pytest

from ingestion.archive import load_archive, season_match_rows
from ingestion.transform import MATCH_COLUMNS


def _season_frames():
    merged = pd.DataFrame({
        "element": [11, 12, 13], "fixture": [1, 1, 1], "GW": [1, 1, 1],
        "kickoff_time": ["2023-08-11T19:00:00Z"] * 3, "team": ["Burnley", "Man City", "Man City"],
        "opponent_team": [2, 1, 1], "was_home": [True, False, False], "position": ["GK", "FWD", "AM"],
        "minutes": [90, 90, 0], "starts": [1, 1, 0], "goals_scored": [0, 2, 0], "assists": [0, 0, 0],
        "expected_goals": [0.0, 1.4, 0.0], "expected_assists": [0.0, 0.1, 0.0],
        "expected_goals_conceded": [2.1, 0.3, 0.0], "clean_sheets": [0, 1, 0], "goals_conceded": [3, 0, 0],
        "own_goals": [0, 0, 0], "penalties_saved": [0, 0, 0], "penalties_missed": [0, 0, 0],
        "saves": [4, 0, 0], "bonus": [0, 3, 0], "yellow_cards": [0, 0, 0], "red_cards": [0, 0, 0],
        "total_points": [1, 13, 0], "value": [45, 140, 0], "xP": [2.0, 7.5, 0.0],
    })
    players_raw = pd.DataFrame({"id": [11, 12, 13], "code": [111, 222, 333]})
    teams = pd.DataFrame({"id": [1, 2], "code": [90, 43], "name": ["Burnley", "Man City"]})
    return merged, players_raw, teams


def test_season_rows_map_codes_positions_and_drop_managers():
    rows = season_match_rows(*_season_frames(), season="2023-24")
    assert list(rows.columns) == MATCH_COLUMNS
    assert len(rows) == 2  # AM row dropped
    gk = rows.set_index("player_code").loc[111]
    assert gk.position == "GKP" and gk.team_code == 90 and gk.opponent_code == 43 and gk.price == 4.5
    assert rows.defensive_contribution.eq(0).all()  # column absent before 2025/26
    assert str(rows.kickoff.dt.tz) == "UTC"


def test_unmapped_team_name_raises():
    merged, players_raw, teams = _season_frames()
    merged.loc[0, "team"] = "Unknown FC"
    with pytest.raises(ValueError, match="Unknown FC"):
        season_match_rows(merged, players_raw, teams, "2023-24")


def test_load_archive_concatenates_seasons():
    frames = dict(zip(["merged", "players_raw", "teams"], _season_frames()))
    log = load_archive(["2022-23", "2023-24"], fetch=lambda s: frames)
    assert sorted(log.season.unique()) == ["2022-23", "2023-24"] and len(log) == 4
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_archive.py -v`
Expected: FAIL (module not found).

- [ ] **Step 4: Implement**

```python
# ingestion/archive.py
"""Past seasons from the public vaastav/Fantasy-Premier-League archive.

The archive's licence is unspecified: data is used for analysis only and never
republished (data/ is not committed). Run once, and again when a season ends:
    python -m ingestion.archive
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from . import storage
from .transform import MATCH_COLUMNS

ARCHIVE_SEASONS = ("2022-23", "2023-24", "2024-25", "2025-26")
ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}"
_FILES = {"merged": "gws/merged_gw.csv", "players_raw": "players_raw.csv", "teams": "teams.csv"}
_RENAME = {
    "GW": "gameweek", "fixture": "fixture_id", "kickoff_time": "kickoff", "goals_scored": "goals",
    "expected_goals": "xg", "expected_assists": "xa", "expected_goals_conceded": "xgc",
    "total_points": "points", "xP": "xp",
}


def fetch_season(season: str, session: requests.Session | None = None) -> dict[str, pd.DataFrame]:
    session = session or requests.Session()
    frames = {}
    for key, path in _FILES.items():
        response = session.get(ARCHIVE_URL.format(season=season, path=path), timeout=60,
                               headers={"User-Agent": "fpl-ai-coach/0.1"})
        response.raise_for_status()
        frames[key] = pd.read_csv(io.BytesIO(response.content), encoding="utf-8", encoding_errors="replace")
    return frames


def season_match_rows(merged: pd.DataFrame, players_raw: pd.DataFrame, teams: pd.DataFrame,
                      season: str) -> pd.DataFrame:
    df = merged[merged["position"] != "AM"].copy()  # 2024/25 Assistant Manager chip rows
    df["position"] = df["position"].replace({"GK": "GKP"})
    df["player_code"] = df["element"].map(players_raw.set_index("id")["code"])
    df["team_code"] = df["team"].map(teams.set_index("name")["code"])
    unmapped = sorted(df.loc[df["team_code"].isna(), "team"].unique())
    if unmapped:
        raise ValueError(f"{season}: team names not in teams.csv: {unmapped}")
    df["opponent_code"] = df["opponent_team"].map(teams.set_index("id")["code"])
    df = df.rename(columns=_RENAME)
    df["season"] = season
    df["price"] = df["value"] / 10
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    if "defensive_contribution" not in df.columns:
        df["defensive_contribution"] = 0
    for col in ("player_code", "team_code", "opponent_code"):
        df[col] = df[col].astype(int)
    return df[MATCH_COLUMNS].reset_index(drop=True)


def load_archive(seasons=ARCHIVE_SEASONS, fetch=fetch_season) -> pd.DataFrame:
    parts = []
    for season in seasons:
        frames = fetch(season)
        parts.append(season_match_rows(frames["merged"], frames["players_raw"], frames["teams"], season))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    log = load_archive()
    path = storage.save_table(log, "archive_match_log")
    print(f"Saved {len(log):,} archive rows ({', '.join(ARCHIVE_SEASONS)}) to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add the live validation test** to `tests/test_live_api.py`

```python
def test_archive_schema_and_scoring_reconstruction():
    """Every archive row's points must be rebuildable from its stats with our scoring rules."""
    from ingestion.archive import ARCHIVE_SEASONS, load_archive
    from prediction.scoring import points_from_stats, rules_for_season

    log = load_archive(ARCHIVE_SEASONS)
    for season, rows in log.groupby("season"):
        exact = (points_from_stats(rows, rules_for_season(season)) == rows["points"]).mean()
        assert exact == 1.0, f"{season}: only {exact:.2%} of rows reconstruct exactly"
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_archive.py -v` then `python -m pytest -m live tests/test_live_api.py::test_archive_schema_and_scoring_reconstruction -v`
Expected: PASS for both (the live test downloads about 20MB).

- [ ] **Step 7: Build the archive file**

Run: `python -m ingestion.archive`
Expected: `Saved ~110,000 archive rows (2022-23, 2023-24, 2024-25, 2025-26) to ...archive_match_log.parquet`; file size a few MB.

- [ ] **Step 8: Commit**

```bash
git add ingestion/archive.py ingestion/transform.py tests/test_archive.py tests/test_live_api.py
git commit -m "Ingestion: past seasons from the public archive as match-log rows" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Current season via `event/{gw}/live` and the combined match log

**Files:**
- Modify: `ingestion/fpl_client.py`, `ingestion/transform.py`, `ingestion/storage.py`, `ingestion/cli.py`
- Test: `tests/test_live_rows.py`; update routes in `tests/test_client_and_cli.py`

**Interfaces:**
- Consumes: `transform.MATCH_COLUMNS`, tables from `ingestion.cli.run` (`players` with `id, code, team, position, price`; `teams` with `id, code`; `fixtures` from `transform.fixtures`; `gameweeks`).
- Produces: `FPLClient.event_live(gameweek) -> dict`; `transform.season_label(gameweeks) -> str` (e.g. `"2026-27"`); `transform.live_match_rows(live, gameweek, fixtures, players, teams, season) -> pd.DataFrame[MATCH_COLUMNS]`; `storage.load_table_or_none(name, data_dir) -> pd.DataFrame | None`; processed tables `current_season_rows` and `match_log`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_live_rows.py
import pandas as pd

from ingestion.transform import MATCH_COLUMNS, live_match_rows, season_label

FIXTURES = pd.DataFrame({
    "id": [1, 2, 3], "event": [6, 6, 6], "team_h": [1, 2, 1], "team_a": [2, 1, 3],
    "kickoff_time": pd.to_datetime(["2026-10-10T14:00Z", "2026-10-12T14:00Z", "2026-10-14T19:00Z"], utc=True),
    "finished_provisional": [True, True, False],
})
PLAYERS = pd.DataFrame({"id": [10, 20], "code": [100, 200], "team": [1, 2],
                        "position": ["MID", "DEF"], "price": [7.0, 5.0]})
TEAMS = pd.DataFrame({"id": [1, 2, 3], "code": [3, 7, 8]})


def _stat(**kw):
    base = {k: 0 for k in ["minutes", "starts", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                           "own_goals", "penalties_saved", "penalties_missed", "saves", "bonus",
                           "defensive_contribution", "yellow_cards", "red_cards", "total_points"]}
    base.update({"expected_goals": "0.00", "expected_assists": "0.00", "expected_goals_conceded": "0.00"})
    base.update(kw)
    return base


LIVE = {"elements": [
    {"id": 10, "stats": _stat(minutes=150, goals_scored=1, expected_goals="0.90", total_points=9),
     "explain": [{"fixture": 1, "stats": [{"identifier": "minutes", "value": 90, "points": 2},
                                          {"identifier": "goals_scored", "value": 1, "points": 5}]},
                 {"fixture": 2, "stats": [{"identifier": "minutes", "value": 60, "points": 2}]}]},
]}


def test_season_label_from_first_deadline():
    gws = pd.DataFrame({"id": [1, 2], "deadline_time": pd.to_datetime(["2026-08-21T17:30Z", "2026-08-28T17:30Z"],
                                                                       utc=True)})
    assert season_label(gws) == "2026-27"


def test_double_gameweek_splits_stats_by_minutes_and_skips_unfinished():
    rows = live_match_rows(LIVE, 6, FIXTURES, PLAYERS, TEAMS, "2026-27")
    assert list(rows.columns) == MATCH_COLUMNS
    mid = rows[rows.player_code == 100].set_index("fixture_id")
    assert list(mid.index) == [1, 2]                           # fixture 3 unfinished: excluded
    assert mid.loc[1, "minutes"] == 90 and mid.loc[2, "minutes"] == 60
    assert mid.loc[1, "points"] == 7 and mid.loc[2, "points"] == 2   # from explain per fixture
    assert abs(mid.loc[1, "xg"] - 0.54) < 1e-9                  # 0.90 x 90/150
    assert mid.loc[1, "opponent_code"] == 7 and bool(mid.loc[1, "was_home"])


def test_player_without_live_entry_gets_zero_minute_rows():
    rows = live_match_rows(LIVE, 6, FIXTURES, PLAYERS, TEAMS, "2026-27")
    defender = rows[rows.player_code == 200]
    assert len(defender) == 2 and defender.minutes.eq(0).all() and defender.xp.isna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_live_rows.py -v`
Expected: FAIL (`live_match_rows` not defined).

- [ ] **Step 3: Implement**

In `ingestion/fpl_client.py`:

```python
    def event_live(self, gameweek: int) -> dict:
        """Every player's stats for one gameweek, with per-fixture points in `explain`."""
        return self.get(f"event/{gameweek}/live")
```

In `ingestion/transform.py`:

```python
_LIVE_STATS = {
    "starts": "starts", "goals": "goals_scored", "assists": "assists", "xg": "expected_goals",
    "xa": "expected_assists", "xgc": "expected_goals_conceded", "clean_sheets": "clean_sheets",
    "goals_conceded": "goals_conceded", "own_goals": "own_goals", "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed", "saves": "saves", "bonus": "bonus",
    "defensive_contribution": "defensive_contribution", "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
}


def season_label(gameweeks: pd.DataFrame) -> str:
    """'2026-27' style label from the first gameweek's deadline."""
    year = int(gameweeks.sort_values("id")["deadline_time"].iloc[0].year)
    return f"{year}-{str(year + 1)[-2:]}"


def live_match_rows(live: dict, gameweek: int, fixtures: pd.DataFrame, players: pd.DataFrame,
                    teams: pd.DataFrame, season: str) -> pd.DataFrame:
    """Match-log rows for one gameweek from `event/{gw}/live`.

    Only finished fixtures are included. In a double gameweek FPL reports one
    combined stat line, so stats are split by the minutes played in each
    fixture (from `explain`), and points come from `explain` per fixture.
    Players with no live entry get zero-minute rows (they didn't play).
    """
    code_of = teams.set_index("id")["code"]
    done = fixtures[(fixtures["event"] == gameweek) & fixtures["finished_provisional"].astype(bool)]
    by_id = {e["id"]: e for e in live.get("elements", [])}
    rows = []
    for p in players.itertuples():
        team_fx = done[(done["team_h"] == p.team) | (done["team_a"] == p.team)]
        if team_fx.empty:
            continue
        entry = by_id.get(p.id, {"stats": {}, "explain": []})
        stats = entry.get("stats", {})
        explain = {x["fixture"]: {s["identifier"]: s for s in x["stats"]} for x in entry.get("explain", [])}
        mins = {f: explain.get(f, {}).get("minutes", {}).get("value", 0) for f in team_fx["id"]}
        total = sum(mins.values())
        single = len(team_fx) == 1
        for fx in team_fx.itertuples():
            share = 1.0 if single else (mins[fx.id] / total if total else 0.0)
            home = fx.team_h == p.team
            row = {
                "season": season, "gameweek": gameweek, "fixture_id": int(fx.id),
                "kickoff": fx.kickoff_time, "player_code": int(p.code),
                "team_code": int(code_of[p.team]),
                "opponent_code": int(code_of[fx.team_a if home else fx.team_h]),
                "was_home": bool(home), "position": p.position, "price": float(p.price), "xp": float("nan"),
                "minutes": float(stats.get("minutes", 0)) if single else float(mins[fx.id]),
                "points": float(stats.get("total_points", 0)) if single
                else float(sum(s["points"] for s in explain.get(fx.id, {}).values())),
            }
            for col, key in _LIVE_STATS.items():
                row[col] = float(stats.get(key, 0) or 0) * share
            rows.append(row)
    return pd.DataFrame(rows, columns=MATCH_COLUMNS)
```

In `ingestion/storage.py`:

```python
def load_table_or_none(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame | None:
    path = data_dir / "processed" / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None
```

In `ingestion/cli.py`, add after the general tables are saved (before the entry pull):

```python
    current_rows = _update_current_season(client, tables, data_dir)
    storage.save_table(current_rows, "current_season_rows", data_dir)
    archive = storage.load_table_or_none("archive_match_log", data_dir)
    if archive is None:
        print("  no archive yet: run `python -m ingestion.archive` for past seasons")
    parts = [x for x in (archive, current_rows) if x is not None and not x.empty]
    if parts:
        match_log = pd.concat(parts, ignore_index=True).sort_values(["kickoff", "fixture_id", "player_code"])
        storage.save_table(match_log, "match_log", data_dir)
```

and the helper:

```python
def _update_current_season(client: FPLClient, tables: dict, data_dir: Path) -> pd.DataFrame:
    """This season's match-log rows. Finished, confirmed gameweeks are fetched once and
    cached; unconfirmed and in-progress gameweeks are re-fetched every run."""
    gws = tables["gameweeks"]
    season = transform.season_label(gws)
    cached = storage.load_table_or_none("current_season_rows", data_dir)
    if cached is not None:
        cached = cached[cached["season"] == season]
    have = set() if cached is None else set(cached["gameweek"])
    todo = [int(g.id) for g in gws[gws["finished"] | gws["is_current"]].itertuples()
            if g.id not in have or not g.data_checked or g.is_current]
    fresh = [transform.live_match_rows(client.event_live(g), g, tables["fixtures"], tables["players"],
                                       tables["teams"], season) for g in todo]
    keep = None if cached is None else cached[~cached["gameweek"].isin(todo)]
    parts = [x for x in (keep, *fresh) if x is not None and not x.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=transform.MATCH_COLUMNS)
```

- [ ] **Step 4: Update existing CLI test routes**

In `tests/test_client_and_cli.py`, the fake API now receives `event/{gw}/live/` calls for finished and current gameweeks (GW1 finished, GW2 current in the fixture). Add to every `routes` dict (and to the two single-route clients in `test_unknown_entry_gives_clear_message_and_keeps_general_tables` and `test_only_latest_raw_snapshot_is_kept`):

```python
        "event/1/live/": {"elements": []},
        "event/2/live/": {"elements": []},
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Live check**

Run: `python -m ingestion.cli --entry 8298351`
Expected: normal summary plus `match_log` and `current_season_rows` in the table list; `current_season_rows` about 667 x 5 rows. A second run fetches only the current gameweek.

- [ ] **Step 7: Commit**

```bash
git add ingestion tests
git commit -m "Ingestion: current season from event/live, combined match log" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Backtest harness with the two benchmarks

**Files:**
- Create: `prediction/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `optimisation.model.solve`, `SquadRules`; match-log columns.
- Produces: `Snapshot` dataclass (`season, gameweek, cutoff, history, players_now, fixtures_ahead, teams_in_season, benchmark_xp`); `snapshot(match_log, season, gameweek, horizon=5) -> Snapshot`; `Predictor = Callable[[Snapshot], pd.DataFrame]` returning columns `player_code, gameweek, total`; `naive_predictor`, `xp_predictor`; `eligible_players(history) -> set[int]`; `actual_points(match_log, season, gameweek) -> pd.Series`; `score_metrics(pred, actual, positions) -> dict`; `decision_points(pred, actual, players_now) -> float`; `run_backtest(match_log, seasons, predictors, horizons=5, decision=True, every=1) -> tuple[pd.DataFrame, pd.DataFrame]`; `summary(per_gw) -> pd.DataFrame`; `calibration(per_player, model, horizon=0) -> pd.DataFrame`; CLI `python -m prediction.backtest`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_backtest.py
import numpy as np
import pandas as pd
import pytest

from ingestion.transform import MATCH_COLUMNS
from prediction.backtest import (actual_points, calibration, decision_points, eligible_players,
                                 naive_predictor, run_backtest, score_metrics, snapshot, xp_predictor)


def synthetic_log(n_gw=8, seed=0):
    """Two-season, 4-team toy league with 6 players per team per position mix."""
    rng = np.random.default_rng(seed)
    rows = []
    teams = [1, 2, 3, 4]
    positions = ["GKP", "DEF", "DEF", "MID", "MID", "FWD"]
    for s_i, season in enumerate(["2022-23", "2023-24"]):
        for gw in range(1, n_gw + 1):
            kickoff = pd.Timestamp("2022-08-06", tz="UTC") + pd.Timedelta(days=365 * s_i + 7 * gw)
            pairs = [(1, 2), (3, 4)] if gw % 2 else [(1, 3), (2, 4)]
            for f_i, (h, a) in enumerate(pairs):
                for team, opp, home in ((h, a, True), (a, h, False)):
                    for k, pos in enumerate(positions):
                        minutes = 90 if k < 5 else int(rng.choice([0, 20, 90]))
                        rows.append({c: 0 for c in MATCH_COLUMNS} | {
                            "season": season, "gameweek": gw, "fixture_id": gw * 10 + f_i, "kickoff": kickoff,
                            "player_code": team * 100 + k, "team_code": team, "opponent_code": opp,
                            "was_home": home, "position": pos, "minutes": minutes,
                            "xg": 0.3 if pos in ("MID", "FWD") and minutes else 0.0,
                            "points": int(rng.integers(0, 10)) if minutes else 0,
                            "price": 5.0 + k, "xp": float(rng.uniform(1, 5)),
                        })
    return pd.DataFrame(rows)


def test_snapshot_hides_the_future():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    assert (snap.history.kickoff < snap.cutoff).all()
    assert snap.fixtures_ahead.gameweek.between(3, 7).all()
    assert set(snap.players_now.columns) >= {"player_code", "team_code", "position", "price", "chance"}


def test_naive_predictor_scales_by_fixture_count():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    pred = naive_predictor(snap)
    code = 100
    last5 = snap.history[snap.history.player_code == code].sort_values("kickoff").tail(5).points.mean()
    assert pred[(pred.player_code == code) & (pred.gameweek == 3)].total.iloc[0] == pytest.approx(last5)


def test_xp_predictor_only_next_gameweek():
    snap = snapshot(synthetic_log(), "2023-24", 3)
    assert set(xp_predictor(snap).gameweek) == {3}


def test_score_metrics_known_values():
    pred = pd.Series([1.0, 2.0, 3.0, 4.0], index=[1, 2, 3, 4])
    actual = pd.Series([2.0, 2.0, 3.0, 6.0], index=[1, 2, 3, 4])
    positions = pd.Series(["MID"] * 4, index=[1, 2, 3, 4])
    m = score_metrics(pred, actual, positions)
    assert m["mae"] == pytest.approx(0.75)
    assert m["rmse"] == pytest.approx(np.sqrt((1 + 0 + 0 + 4) / 4))
    assert m["rho"] == pytest.approx(0.9486832980505138)


def test_eligible_and_actual_points():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 3)
    assert 100 in eligible_players(snap.history)
    actual = actual_points(log, "2023-24", 3)
    assert actual.index.name == "player_code" and actual.ge(0).all()


def test_decision_points_is_the_real_xi_score():
    from tests.optimiser_helpers import full_pool, pool
    players, scores = pool(full_pool())
    players_now = players.rename(columns={"id": "player_code", "team": "team_code"})
    actual = pd.Series(1.0, index=players_now.player_code)
    value = decision_points(scores.rename_axis("player_code"), actual, players_now)
    assert value == pytest.approx(12.0)  # 11 starters x 1 + captain bonus 1


def test_run_backtest_and_calibration():
    log = synthetic_log()
    per_gw, per_player = run_backtest(log, ["2023-24"], {"naive": naive_predictor, "xp": xp_predictor},
                                      horizons=2, decision=False)
    assert set(per_gw.model) == {"naive", "xp"}
    assert set(per_gw.loc[per_gw.model == "xp", "horizon"]) == {0}
    table = calibration(per_player, "naive")
    assert list(table.columns) == ["mean_predicted", "mean_actual", "players"]
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# prediction/backtest.py
"""Walk-forward backtest: replay past gameweeks using only what was known at each deadline.

    python -m prediction.backtest --seasons 2022-23,2023-24,2024-25 --models naive,xp
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ingestion import storage
from optimisation.model import SquadRules, solve

STANDARD_RULES = SquadRules(
    composition={"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3},
    xi_min={"GKP": 1, "DEF": 3, "MID": 2, "FWD": 1},
    xi_max={"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3},
    starting_xi=11, max_per_club=3, hit_cost=4, max_free_transfers=5,
)
WARM_UP_GAMEWEEKS = 5


@dataclass
class Snapshot:
    """Everything knowable at a gameweek's deadline."""
    season: str
    gameweek: int
    cutoff: pd.Timestamp
    history: pd.DataFrame        # match-log rows with kickoff < cutoff
    players_now: pd.DataFrame    # player_code, team_code, position, price, chance
    fixtures_ahead: pd.DataFrame  # gameweek, fixture_id, kickoff, home_code, away_code
    teams_in_season: list[int]
    benchmark_xp: pd.Series      # FPL's xP for this gameweek (published before the deadline)


Predictor = Callable[[Snapshot], pd.DataFrame]


def snapshot(match_log: pd.DataFrame, season: str, gameweek: int, horizon: int = 5) -> Snapshot:
    s = match_log[match_log["season"] == season]
    cutoff = s.loc[s["gameweek"] == gameweek, "kickoff"].min()
    now = s[s["gameweek"] == gameweek].sort_values("kickoff").drop_duplicates("player_code")
    ahead = s[(s["gameweek"] >= gameweek) & (s["gameweek"] < gameweek + horizon) & s["was_home"]]
    fixtures = (ahead[["gameweek", "fixture_id", "kickoff", "team_code", "opponent_code"]]
                .drop_duplicates("fixture_id")
                .rename(columns={"team_code": "home_code", "opponent_code": "away_code"}))
    xp = s[s["gameweek"] == gameweek].groupby("player_code")["xp"].sum()
    return Snapshot(
        season=season, gameweek=gameweek, cutoff=cutoff,
        history=match_log[match_log["kickoff"] < cutoff],
        players_now=now[["player_code", "team_code", "position", "price"]].assign(chance=np.nan)
        .reset_index(drop=True),
        fixtures_ahead=fixtures.reset_index(drop=True),
        teams_in_season=sorted(s["team_code"].unique()),
        benchmark_xp=xp,
    )


def _team_fixture_counts(fixtures_ahead: pd.DataFrame) -> pd.DataFrame:
    sides = pd.concat([fixtures_ahead[["gameweek", "home_code"]].rename(columns={"home_code": "team_code"}),
                       fixtures_ahead[["gameweek", "away_code"]].rename(columns={"away_code": "team_code"})])
    return sides.groupby(["team_code", "gameweek"]).size().rename("n_fixtures").reset_index()


def naive_predictor(snap: Snapshot) -> pd.DataFrame:
    """Mean points over the player's last 5 matches, times fixtures in each gameweek."""
    last5 = snap.history.sort_values("kickoff").groupby("player_code").tail(5)
    per_match = last5.groupby("player_code")["points"].mean().rename("per_match")
    p = snap.players_now.join(per_match, on="player_code").fillna({"per_match": 0.0})
    rows = p.merge(_team_fixture_counts(snap.fixtures_ahead), on="team_code")
    rows["total"] = rows["per_match"] * rows["n_fixtures"]
    return rows[["player_code", "gameweek", "total"]]


def xp_predictor(snap: Snapshot) -> pd.DataFrame:
    """FPL's own expected points; exists for the next gameweek only."""
    xp = snap.benchmark_xp.rename("total").reset_index()
    return xp.assign(gameweek=snap.gameweek)[["player_code", "gameweek", "total"]]


def eligible_players(history: pd.DataFrame) -> set[int]:
    last5 = history.sort_values("kickoff").groupby("player_code").tail(5)
    return set(last5.loc[last5["minutes"] > 0, "player_code"])


def actual_points(match_log: pd.DataFrame, season: str, gameweek: int) -> pd.Series:
    rows = match_log[(match_log["season"] == season) & (match_log["gameweek"] == gameweek)]
    return rows.groupby("player_code")["points"].sum()


def score_metrics(pred: pd.Series, actual: pd.Series, positions: pd.Series) -> dict:
    err = pred - actual
    rhos = []
    for _, idx in positions.groupby(positions).groups.items():
        if len(idx) >= 5 or len(positions.unique()) == 1:
            rho = spearmanr(pred[idx], actual[idx]).statistic
            if not np.isnan(rho):
                rhos.append(rho)
    return {"mae": float(err.abs().mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "rho": float(np.mean(rhos)) if rhos else float("nan"), "n": int(len(err))}


def decision_points(pred: pd.Series, actual: pd.Series, players_now: pd.DataFrame) -> float:
    """Realised points of the XI and captain the optimiser picks from `pred` (from scratch, £100m)."""
    players = players_now.rename(columns={"player_code": "id", "team_code": "team"})
    players = players.assign(web_name=players["id"].astype(str))
    scores = pred.clip(lower=0)
    sol = solve(players, scores, STANDARD_RULES, budget=100.0, bench_weight=0.1)
    got = lambda i: float(actual.get(i, 0.0))  # noqa: E731
    return sum(got(i) for i in sol.starting) + got(sol.captain)


def run_backtest(match_log: pd.DataFrame, seasons: list[str], predictors: dict[str, Predictor],
                 horizons: int = 5, decision: bool = True, every: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_season = match_log["season"].min()
    per_gw, per_player = [], []
    for season in seasons:
        gameweeks = sorted(match_log.loc[match_log["season"] == season, "gameweek"].unique())
        for gw in gameweeks[::every]:
            if season == first_season and gw <= WARM_UP_GAMEWEEKS:
                continue
            snap = snapshot(match_log, season, gw, horizons)
            eligible = eligible_players(snap.history)
            positions = snap.players_now.set_index("player_code")["position"]
            for name, predictor in predictors.items():
                pred = predictor(snap)
                for h in range(horizons):
                    target = gw + h
                    p = pred[pred["gameweek"] == target].groupby("player_code")["total"].sum()
                    if p.empty:
                        continue
                    actual = actual_points(match_log, season, target)
                    idx = sorted(eligible & set(positions.index))
                    p_e = p.reindex(idx).fillna(0.0)
                    a_e = actual.reindex(idx).fillna(0.0)
                    row = {"model": name, "season": season, "gameweek": gw, "horizon": h,
                           **score_metrics(p_e, a_e, positions.reindex(idx))}
                    if decision and h == 0:
                        row["decision"] = decision_points(p, actual, snap.players_now)
                    per_gw.append(row)
                    per_player.append(pd.DataFrame({"model": name, "season": season, "gameweek": gw,
                                                    "horizon": h, "player_code": idx,
                                                    "predicted": p_e.to_numpy(), "actual": a_e.to_numpy()}))
    return pd.DataFrame(per_gw), pd.concat(per_player, ignore_index=True)


def summary(per_gw: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("mae", "rmse", "rho", "decision") if c in per_gw.columns]
    return per_gw.groupby(["model", "horizon"])[cols].mean().round(3)


def calibration(per_player: pd.DataFrame, model: str, horizon: int = 0) -> pd.DataFrame:
    rows = per_player[(per_player["model"] == model) & (per_player["horizon"] == horizon)]
    groups = pd.qcut(rows["predicted"].rank(method="first"), 10, labels=False)
    return rows.groupby(groups).agg(mean_predicted=("predicted", "mean"), mean_actual=("actual", "mean"),
                                    players=("actual", "size")).round(2)


def build_predictors(names: list[str], params_path: str | None = None) -> dict[str, Predictor]:
    predictors: dict[str, Predictor] = {"naive": naive_predictor, "xp": xp_predictor}
    return {n: predictors[n] for n in names}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of prediction models.")
    parser.add_argument("--seasons", default="2022-23,2023-24,2024-25")
    parser.add_argument("--models", default="naive,xp")
    parser.add_argument("--params", help="model_params.json for the component model")
    parser.add_argument("--every", type=int, default=1, help="use every Nth gameweek (faster)")
    parser.add_argument("--no-decision", action="store_true")
    args = parser.parse_args(argv)
    log = storage.load_table("archive_match_log")
    per_gw, per_player = run_backtest(log, args.seasons.split(","),
                                      build_predictors(args.models.split(","), args.params),
                                      decision=not args.no_decision, every=args.every)
    print(summary(per_gw).to_string())
    for model in args.models.split(","):
        print(f"\nCalibration ({model}, next gameweek):\n{calibration(per_player, model).to_string()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_backtest.py -v`
Expected: PASS. If `test_decision_points_is_the_real_xi_score` fails on the captain arithmetic, check the expected value: 11 starters at 1 point plus the captain's extra 1 = 12.

- [ ] **Step 5: Checkpoint 0: the numbers to beat**

Run: `python -m prediction.backtest --seasons 2022-23,2023-24,2024-25 --models naive,xp`
Expected: a summary table (MAE, RMSE, rho, decision) for `naive` at horizons 0 to 4 and `xp` at horizon 0, plus calibration tables. **Paste the table into the task report**; these are the benchmarks the component model must beat.

- [ ] **Step 6: Commit**

```bash
git add prediction/backtest.py tests/test_backtest.py
git commit -m "Prediction: walk-forward backtest with naive and FPL xP benchmarks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Team ratings

**Files:**
- Create: `features/__init__.py`, `features/team_ratings.py`
- Modify: `pyproject.toml` (add `"scipy>=1.11"` to dependencies; add `"features"` to `[tool.setuptools] packages`)
- Test: `tests/test_team_ratings.py`

**Interfaces:**
- Consumes: match-log columns.
- Produces: `team_matches(match_log) -> pd.DataFrame` (`season, fixture_id, kickoff, team_code, opponent_code, was_home, xg, goals`); `TeamParams` (frozen dataclass: `half_life_days=180.0, ridge=2.0, xg_weight=1.0, prev_season_fade=0.5, promoted_attack=-0.15, promoted_defence=-0.15`); `TeamRatings` (dataclass: `attack: dict[int, float], defence: dict[int, float], mu: float, home: float, unknown_attack: float, unknown_defence: float`; method `expected_goals(team_codes, opponent_codes, is_home) -> np.ndarray`); `fit_team_ratings(tm, cutoff, season, params, teams_in_season) -> TeamRatings`.

- [ ] **Step 1: Install scipy**

Run: `python -m pip install "scipy>=1.11"` then `python -c "import scipy; print(scipy.__version__)"`
Expected: a version prints. Add `"scipy>=1.11",` to `pyproject.toml` dependencies and `"features"` to the packages list.

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_team_ratings.py
import numpy as np
import pandas as pd
import pytest

from features.team_ratings import TeamParams, fit_team_ratings, team_matches


def simulate(true_a, true_d, mu=0.3, home=0.25, seasons=("2022-23", "2023-24"), seed=1):
    rng = np.random.default_rng(seed)
    teams = list(range(len(true_a)))
    rows, fid = [], 0
    start = pd.Timestamp("2022-08-01", tz="UTC")
    for s_i, season in enumerate(seasons):
        day = 0
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                fid += 1
                day += 1
                kickoff = start + pd.Timedelta(days=365 * s_i + day // 5)
                lam_h = np.exp(mu + home + true_a[h] - true_d[a])
                lam_a = np.exp(mu + true_a[a] - true_d[h])
                for team, opp, is_home, lam in ((h, a, True, lam_h), (a, h, False, lam_a)):
                    g = rng.poisson(lam)
                    rows.append({"season": season, "fixture_id": fid, "kickoff": kickoff, "team_code": team,
                                 "opponent_code": opp, "was_home": is_home, "xg": lam, "goals": g})
    return pd.DataFrame(rows)


def test_parameter_recovery_from_simulated_seasons():
    rng = np.random.default_rng(0)
    true_a = rng.normal(0, 0.3, 20); true_a -= true_a.mean()
    true_d = rng.normal(0, 0.3, 20); true_d -= true_d.mean()
    tm = simulate(true_a, true_d)
    params = TeamParams(half_life_days=1e6, ridge=0.01, prev_season_fade=1.0, xg_weight=0.0)
    r = fit_team_ratings(tm, tm.kickoff.max() + pd.Timedelta(days=1), "2023-24", params, list(range(20)))
    a = np.array([r.attack[t] for t in range(20)]); d = np.array([r.defence[t] for t in range(20)])
    assert np.corrcoef(a, true_a)[0, 1] > 0.9 and np.corrcoef(d, true_d)[0, 1] > 0.9
    assert r.home == pytest.approx(0.25, abs=0.08)


def test_fit_ignores_matches_after_cutoff():
    tm = simulate(np.zeros(4), np.zeros(4))
    cutoff = tm.kickoff.iloc[len(tm) // 2]
    future = tm[tm.kickoff >= cutoff].assign(xg=9.0, goals=9)
    base = fit_team_ratings(tm[tm.kickoff < cutoff], cutoff, "2023-24", TeamParams(), [0, 1, 2, 3])
    leaky = fit_team_ratings(pd.concat([tm, future]), cutoff, "2023-24", TeamParams(), [0, 1, 2, 3])
    assert base.attack == pytest.approx(leaky.attack) and base.mu == pytest.approx(leaky.mu)


def test_promoted_team_gets_prior_and_finite_goals():
    tm = simulate(np.zeros(4), np.zeros(4), seasons=("2022-23",))
    cutoff = pd.Timestamp("2023-08-01", tz="UTC")
    params = TeamParams(promoted_attack=-0.2, promoted_defence=-0.2)
    r = fit_team_ratings(tm, cutoff, "2023-24", params, teams_in_season=[0, 1, 2, 99])
    assert r.attack[99] == pytest.approx(-0.2) and r.defence[99] == pytest.approx(-0.2)
    lam = r.expected_goals(np.array([99]), np.array([0]), np.array([True]))
    assert np.isfinite(lam).all() and lam[0] > 0


def test_team_matches_counts_opponent_own_goals():
    log = pd.DataFrame({
        "season": ["2023-24"] * 2, "fixture_id": [1, 1], "kickoff": pd.to_datetime(["2023-08-11"] * 2, utc=True),
        "team_code": [3, 7], "opponent_code": [7, 3], "was_home": [True, False],
        "xg": [1.2, 0.8], "goals": [1, 0], "own_goals": [0, 1],
    })
    tm = team_matches(log).set_index("team_code")
    assert tm.loc[3, "goals"] == 2 and tm.loc[7, "goals"] == 0 and tm.loc[3, "xg"] == pytest.approx(1.2)
```

- [ ] **Step 3: Run to verify failure**

Run: `python -m pytest tests/test_team_ratings.py -v`
Expected: FAIL (module not found).

- [ ] **Step 4: Implement**

```python
# features/__init__.py
"""Feature layer: cutoff-aware features computed from the match log."""
```

```python
# features/team_ratings.py
"""Team attack/defence ratings from a Poisson model fitted to xG.

For a match where H hosts A:
    lambda_H = exp(mu + home + a_H - d_A)
    lambda_A = exp(mu        + a_A - d_H)
Fitted by weighted Poisson maximum likelihood with a ridge penalty pulling each
team towards its prior (0, or the promoted-team prior). Only matches with
kickoff < cutoff are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

MIN_WEIGHT = 1e-4


@dataclass(frozen=True)
class TeamParams:
    half_life_days: float = 180.0
    ridge: float = 2.0
    xg_weight: float = 1.0          # alpha: target = alpha*xG + (1-alpha)*goals
    prev_season_fade: float = 0.5   # extra weight multiplier per season back
    promoted_attack: float = -0.15
    promoted_defence: float = -0.15


@dataclass
class TeamRatings:
    attack: dict[int, float]
    defence: dict[int, float]
    mu: float
    home: float
    unknown_attack: float
    unknown_defence: float

    def expected_goals(self, team_codes, opponent_codes, is_home) -> np.ndarray:
        a = pd.Series(np.asarray(team_codes)).map(self.attack).fillna(self.unknown_attack).to_numpy()
        d = pd.Series(np.asarray(opponent_codes)).map(self.defence).fillna(self.unknown_defence).to_numpy()
        return np.exp(self.mu + self.home * np.asarray(is_home, dtype=float) + a - d)


def team_matches(match_log: pd.DataFrame) -> pd.DataFrame:
    """One row per team per fixture: xG (sum of players' xG) and goals (own + opponent own goals)."""
    keys = ["season", "fixture_id", "kickoff", "team_code", "opponent_code", "was_home"]
    tm = match_log.groupby(keys, as_index=False).agg(xg=("xg", "sum"), goals=("goals", "sum"),
                                                     own_goals=("own_goals", "sum"))
    opp_og = tm[["season", "fixture_id", "team_code", "own_goals"]].rename(
        columns={"team_code": "opponent_code", "own_goals": "opp_own_goals"})
    tm = tm.merge(opp_og, on=["season", "fixture_id", "opponent_code"], how="left")
    tm["goals"] = tm["goals"] + tm["opp_own_goals"].fillna(0)
    return tm.drop(columns=["own_goals", "opp_own_goals"])


def _season_weights(data: pd.DataFrame, cutoff: pd.Timestamp, season: str, params: TeamParams) -> np.ndarray:
    order = {s: i for i, s in enumerate(sorted(set(data["season"]) | {season}))}
    back = order[season] - data["season"].map(order)
    age_days = (cutoff - data["kickoff"]).dt.total_seconds() / 86400
    return (0.5 ** (age_days / params.half_life_days) * params.prev_season_fade ** back).to_numpy()


def fit_team_ratings(tm: pd.DataFrame, cutoff: pd.Timestamp, season: str, params: TeamParams,
                     teams_in_season: list[int]) -> TeamRatings:
    data = tm[tm["kickoff"] < cutoff]
    seasons = sorted(set(tm["season"]) | {season})
    prev = seasons[seasons.index(season) - 1] if seasons.index(season) > 0 else None
    prev_teams = set(tm.loc[tm["season"] == prev, "team_code"]) if prev else None
    promoted = {t for t in teams_in_season if prev_teams is not None and t not in prev_teams}

    w = _season_weights(data, cutoff, season, params) if len(data) else np.array([])
    keep = w > MIN_WEIGHT
    data, w = data[keep], w[keep]
    teams = sorted(set(data["team_code"]) | set(data["opponent_code"]) | set(teams_in_season))
    n = len(teams)
    pos = {t: i for i, t in enumerate(teams)}
    prior_a = np.array([params.promoted_attack if t in promoted else 0.0 for t in teams])
    prior_d = np.array([params.promoted_defence if t in promoted else 0.0 for t in teams])

    if len(data) == 0:  # nothing to fit: everyone sits at their prior
        return TeamRatings(dict(zip(teams, prior_a)), dict(zip(teams, prior_d)), float(np.log(1.35)), 0.2,
                           params.promoted_attack, params.promoted_defence)

    it = data["team_code"].map(pos).to_numpy()
    io = data["opponent_code"].map(pos).to_numpy()
    h = data["was_home"].astype(float).to_numpy()
    y = (params.xg_weight * data["xg"] + (1 - params.xg_weight) * data["goals"]).to_numpy(dtype=float)

    def objective(theta):
        mu, home, a, d = theta[0], theta[1], theta[2:2 + n], theta[2 + n:]
        eta = mu + home * h + a[it] - d[io]
        lam = np.exp(eta)
        f = np.sum(w * (lam - y * eta)) + params.ridge * (np.sum((a - prior_a) ** 2) + np.sum((d - prior_d) ** 2))
        r = w * (lam - y)
        grad = np.concatenate([
            [r.sum(), (r * h).sum()],
            np.bincount(it, r, n) + 2 * params.ridge * (a - prior_a),
            -np.bincount(io, r, n) + 2 * params.ridge * (d - prior_d),
        ])
        return f, grad

    theta0 = np.concatenate([[np.log(max(np.average(y, weights=w), 0.1)), 0.2], prior_a, prior_d])
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Team rating fit did not converge: {result.message}")
    t = result.x
    return TeamRatings(dict(zip(teams, t[2:2 + n])), dict(zip(teams, t[2 + n:])), float(t[0]), float(t[1]),
                       params.promoted_attack, params.promoted_defence)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_team_ratings.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add features pyproject.toml tests/test_team_ratings.py
git commit -m "Features: Poisson team ratings fitted to xG, with recovery and leakage tests" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Player features (minutes, rates, priors)

**Files:**
- Create: `features/player_rates.py`
- Test: `tests/test_player_rates.py`

**Interfaces:**
- Consumes: `features.team_ratings.TeamRatings`, `prediction.scoring.DC_FIRST_SEASON`, `ScoringRules().dc_threshold` values (DEF 10, MID/FWD 12).
- Produces: `PlayerParams` (frozen dataclass: `half_life_days=120.0, prev_season_fade=0.5, kappa_minutes=3.0, kappa_typical_minutes=3.0, kappa_xg=4.0, kappa_xa=4.0, kappa_bonus=6.0, kappa_saves=6.0, kappa_dc=5.0, kappa_cards=10.0`); `PRICE_BANDS = (5.5, 7.5, 10.0)`; `price_band(prices) -> np.ndarray`; `RATE_COLUMNS = ["p60", "psub", "m60", "msub", "xg_rel", "xa_rel", "bonus90", "saves_rel", "p_dc", "yellow90", "red90"]`; `player_features(history, cutoff, season, ratings, params) -> tuple[pd.DataFrame, pd.DataFrame]` where the first frame is indexed by `player_code` with `RATE_COLUMNS` plus `team_code, position, price, recent_minutes`, and the second is priors indexed by `(position, band)` with `RATE_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_player_rates.py
import numpy as np
import pandas as pd
import pytest

from features.player_rates import PlayerParams, player_features, price_band
from features.team_ratings import TeamRatings
from tests.test_backtest import synthetic_log

FLAT = TeamRatings({}, {}, mu=0.0, home=0.0, unknown_attack=0.0, unknown_defence=0.0)  # every lambda = 1
CUTOFF = pd.Timestamp("2023-09-01", tz="UTC")


def test_price_bands():
    assert list(price_band(np.array([4.5, 5.5, 7.4, 9.9, 12.0]))) == [0, 1, 1, 2, 3]


def test_ignores_matches_after_cutoff():
    log = synthetic_log()
    future = log[log.kickoff >= CUTOFF].assign(minutes=90, xg=5.0)
    a, _ = player_features(log[log.kickoff < CUTOFF], CUTOFF, "2023-24", FLAT, PlayerParams())
    b, _ = player_features(pd.concat([log, future]), CUTOFF, "2023-24", FLAT, PlayerParams())
    pd.testing.assert_frame_equal(a, b)


def test_shrinkage_limits():
    log = synthetic_log()
    none, priors = player_features(log, CUTOFF, "2023-24", FLAT, PlayerParams(kappa_xg=1e9))
    mid = none.loc[103]
    assert mid.xg_rel == pytest.approx(priors.loc[(mid.position, int(price_band(np.array([mid.price]))[0])),
                                                  "xg_rel"])
    raw, _ = player_features(log, CUTOFF, "2023-24", FLAT, PlayerParams(kappa_xg=1e-9))
    assert raw.loc[103].xg_rel == pytest.approx(0.3, rel=1e-6)  # 0.3 xG per 90 at lambda 1


def test_regular_starter_has_high_p60():
    feats, _ = player_features(synthetic_log(), CUTOFF, "2023-24", FLAT, PlayerParams(kappa_minutes=0.1))
    assert feats.loc[100].p60 > 0.95 and feats.loc[100].psub < 0.05


def test_transferred_player_uses_latest_team_and_context():
    log = synthetic_log()
    moved = log.player_code == 103
    late = moved & (log.kickoff > pd.Timestamp("2023-08-20", tz="UTC"))
    log.loc[late, "team_code"] = 2
    feats, _ = player_features(log, CUTOFF, "2023-24", FLAT, PlayerParams())
    assert feats.loc[103].team_code == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_player_rates.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# features/player_rates.py
"""Per-player minutes and per-90 rates with recency weighting and shrinkage.

Every rate is a weighted sum over the player's matches before the cutoff:
    weight = 0.5^(age_days / half_life) * prev_season_fade^(seasons back)
shrunk towards the average for his position and price band:
    shrunk = (sum_w_x + kappa * prior) / (sum_w_exposure + kappa)
Attacking rates are stored relative to the team's expected goals (lambda) in
the matches where they were earned, so they transfer between teams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from prediction.scoring import DC_FIRST_SEASON

from .team_ratings import TeamRatings

PRICE_BANDS = (5.5, 7.5, 10.0)
DC_THRESHOLD = {"DEF": 10, "MID": 12, "FWD": 12}
RATE_COLUMNS = ["p60", "psub", "m60", "msub", "xg_rel", "xa_rel", "bonus90", "saves_rel", "p_dc",
                "yellow90", "red90"]
TYPICAL_MINUTES = {"m60": 87.0, "msub": 25.0}


@dataclass(frozen=True)
class PlayerParams:
    half_life_days: float = 120.0
    prev_season_fade: float = 0.5
    kappa_minutes: float = 3.0
    kappa_typical_minutes: float = 3.0
    kappa_xg: float = 4.0
    kappa_xa: float = 4.0
    kappa_bonus: float = 6.0
    kappa_saves: float = 6.0
    kappa_dc: float = 5.0
    kappa_cards: float = 10.0


def price_band(prices) -> np.ndarray:
    return np.digitize(np.asarray(prices, dtype=float), PRICE_BANDS)


# (numerator sum, denominator sum, kappa field) for each rate
_RATES = {
    "p60": ("W60", "W", "kappa_minutes"),
    "psub": ("Wsub", "W", "kappa_minutes"),
    "m60": ("M60", "W60", "kappa_typical_minutes"),
    "msub": ("MS", "Wsub", "kappa_typical_minutes"),
    "xg_rel": ("XG", "LT", "kappa_xg"),
    "xa_rel": ("XA", "LT", "kappa_xa"),
    "bonus90": ("B", "E", "kappa_bonus"),
    "saves_rel": ("S", "LO", "kappa_saves"),
    "p_dc": ("DC", "DCW", "kappa_dc"),
    "yellow90": ("Y", "E", "kappa_cards"),
    "red90": ("R", "E", "kappa_cards"),
}


def _weighted_sums(h: pd.DataFrame, ratings: TeamRatings) -> pd.DataFrame:
    w = h["w"]
    s60 = (h["minutes"] >= 60).astype(float)
    ssub = ((h["minutes"] > 0) & (h["minutes"] < 60)).astype(float)
    e90 = h["minutes"] / 90
    lam_team = ratings.expected_goals(h["team_code"], h["opponent_code"], h["was_home"])
    lam_opp = ratings.expected_goals(h["opponent_code"], h["team_code"], ~h["was_home"].astype(bool))
    dc_valid = (h["season"] >= DC_FIRST_SEASON) & (h["position"] != "GKP")
    dc_hit = (h["defensive_contribution"] >= h["position"].map(DC_THRESHOLD).fillna(np.inf)).astype(float)
    return pd.DataFrame({
        "W": w, "W60": w * s60, "Wsub": w * ssub, "E": w * e90,
        "M60": w * s60 * h["minutes"], "MS": w * ssub * h["minutes"],
        "XG": w * h["xg"], "XA": w * h["xa"], "LT": w * e90 * lam_team,
        "B": w * h["bonus"], "S": w * h["saves"], "LO": w * e90 * lam_opp,
        "DC": w * s60 * dc_hit * dc_valid, "DCW": w * s60 * dc_valid,
        "Y": w * h["yellow_cards"], "R": w * h["red_cards"],
    }, index=h.index)


def player_features(history: pd.DataFrame, cutoff: pd.Timestamp, season: str, ratings: TeamRatings,
                    params: PlayerParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = history[history["kickoff"] < cutoff].copy()
    order = {s: i for i, s in enumerate(sorted(set(h["season"]) | {season}))}
    back = order[season] - h["season"].map(order)
    age = (cutoff - h["kickoff"]).dt.total_seconds() / 86400
    h["w"] = 0.5 ** (age / params.half_life_days) * params.prev_season_fade ** back
    h["band"] = price_band(h["price"])
    sums = _weighted_sums(h, ratings)

    # Priors: same weighted sums pooled by (position, band); fall back to position, then to typical values.
    by_group = sums.groupby([h["position"], h["band"]]).sum()
    by_pos = sums.groupby(h["position"]).sum()
    priors = pd.DataFrame(index=by_group.index)
    for col, (num, den, _) in _RATES.items():
        group_rate = by_group[num] / by_group[den].replace(0, np.nan)
        pos_rate = (by_pos[num] / by_pos[den].replace(0, np.nan)).reindex(by_group.index.get_level_values(0))
        priors[col] = group_rate.fillna(pd.Series(pos_rate.to_numpy(), index=by_group.index))
    for col, typical in TYPICAL_MINUTES.items():
        priors[col] = priors[col].fillna(typical)
    priors = priors.fillna(0.0)

    per_player = sums.groupby(h["player_code"]).sum()
    last = h.sort_values("kickoff").groupby("player_code").tail(1).set_index("player_code")
    recent = h.sort_values("kickoff").groupby("player_code").tail(5).groupby("player_code")["minutes"].sum()
    bands = price_band(last["price"])
    prior_rows = priors.reindex(pd.MultiIndex.from_arrays([last["position"], bands])).set_axis(last.index)

    feats = pd.DataFrame(index=per_player.index)
    for col, (num, den, kappa_field) in _RATES.items():
        kappa = getattr(params, kappa_field)
        prior = prior_rows[col].reindex(per_player.index).fillna(priors[col].mean())
        feats[col] = (per_player[num] + kappa * prior) / (per_player[den] + kappa)
    feats["team_code"] = last["team_code"].reindex(feats.index)
    feats["position"] = last["position"].reindex(feats.index)
    feats["price"] = last["price"].reindex(feats.index)
    feats["recent_minutes"] = recent.reindex(feats.index)
    return feats.sort_index(), priors
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_player_rates.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add features/player_rates.py tests/test_player_rates.py
git commit -m "Features: shrunk player minutes and per-90 rates in team context" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Poisson helpers and the component model

**Files:**
- Create: `prediction/poisson.py`, `prediction/component_model.py`
- Test: `tests/test_component_model.py`

**Interfaces:**
- Consumes: `fit_team_ratings`, `team_matches`, `TeamParams` (Task 6); `player_features`, `PlayerParams`, `RATE_COLUMNS`, `price_band` (Task 7); `rules_for_season` (Task 2); `backtest.Snapshot` fields (Task 5).
- Produces: `expected_floor_div(lam, k) -> np.ndarray`; `COMPONENTS = ("appearance", "goals", "assists", "clean_sheet", "conceded", "saves", "bonus", "defensive", "discipline")`; `ModelParams` (frozen dataclass: `team: TeamParams, player: PlayerParams, flag_fade: float = 0.5`; `to_dict()`, `ModelParams.from_dict(d)`); `fixture_components(rows, rules, components=COMPONENTS) -> pd.DataFrame`; `predict(history, players_now, fixtures_ahead, cutoff, season, teams_in_season, params, components=COMPONENTS) -> pd.DataFrame` (columns `player_code, gameweek, fixture_id, horizon, opponent_code, was_home, no_history, p60, psub, exp_minutes, <each component>, total`); `per_gameweek(pred) -> pd.DataFrame` (`player_code, gameweek, total`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_component_model.py
import math

import numpy as np
import pandas as pd
import pytest

from prediction.component_model import ModelParams, fixture_components, per_gameweek, predict
from prediction.poisson import expected_floor_div
from prediction.scoring import rules_for_season
from tests.test_backtest import synthetic_log


def _brute(lam, k):
    return sum((n // k) * math.exp(-lam) * lam ** n / math.factorial(n) for n in range(80))


@pytest.mark.parametrize("lam,k", [(0.3, 2), (1.0, 2), (2.7, 2), (4.0, 3), (7.5, 3)])
def test_expected_floor_div_matches_brute_force(lam, k):
    assert expected_floor_div(np.array([lam]), k)[0] == pytest.approx(_brute(lam, k), rel=1e-9)


def _row(**kw):
    base = dict(position="DEF", p60=0.8, psub=0.1, m60=90.0, msub=30.0, xg_rel=0.1, xa_rel=0.05,
                bonus90=0.3, saves_rel=0.0, p_dc=0.4, yellow90=0.1, red90=0.0, lam_team=1.5, lam_opp=1.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_one_fixture_by_hand():
    c = fixture_components(_row(), rules_for_season("2025-26")).iloc[0]
    em = 0.8 * 90 + 0.1 * 30  # expected minutes = 75
    assert c.appearance == pytest.approx(2 * 0.8 + 0.1)
    assert c.goals == pytest.approx(6 * 0.1 * 1.5 * em / 90)
    assert c.assists == pytest.approx(3 * 0.05 * 1.5 * em / 90)
    assert c.clean_sheet == pytest.approx(4 * 0.8 * math.exp(-1.0))
    assert c.conceded == pytest.approx(-0.8 * _brute(1.0, 2))
    assert c.saves == 0.0
    assert c.bonus == pytest.approx(0.3 * em / 90)
    assert c.defensive == pytest.approx(2 * 0.8 * 0.4)
    assert c.discipline == pytest.approx(-1 * 0.1 * em / 90)
    assert c.total == pytest.approx(sum(c[k] for k in ["appearance", "goals", "assists", "clean_sheet",
                                                       "conceded", "saves", "bonus", "defensive", "discipline"]))


def test_no_defensive_contribution_before_2025_26():
    assert fixture_components(_row(), rules_for_season("2024-25")).iloc[0].defensive == 0.0


def test_goalkeeper_saves_and_component_switch():
    gk = _row(position="GKP", saves_rel=3.0, p_dc=0.0)
    c = fixture_components(gk, rules_for_season("2024-25")).iloc[0]
    assert c.saves == pytest.approx(0.8 * _brute(3.0 * 1.0 * 90 / 90, 3))
    only = fixture_components(gk, rules_for_season("2024-25"), components=("appearance",)).iloc[0]
    assert only.total == pytest.approx(only.appearance) and only.saves == 0.0


def _snapshot_inputs(log, gameweek=3):
    from prediction.backtest import snapshot
    return snapshot(log, "2023-24", gameweek)


def test_double_gameweek_sums_and_blank_gives_nothing():
    snap = _snapshot_inputs(synthetic_log())
    fx = snap.fixtures_ahead
    first = fx[fx.gameweek == 3].iloc[0]
    double = pd.concat([fx, fx[fx.gameweek == 3].iloc[[0]].assign(fixture_id=999)], ignore_index=True)
    blank = fx[fx.gameweek != 3]
    params = ModelParams()
    kw = dict(history=snap.history, players_now=snap.players_now, cutoff=snap.cutoff, season="2023-24",
              teams_in_season=snap.teams_in_season, params=params)
    single = per_gameweek(predict(fixtures_ahead=fx, **kw)).set_index(["player_code", "gameweek"]).total
    doubled = per_gameweek(predict(fixtures_ahead=double, **kw)).set_index(["player_code", "gameweek"]).total
    code = snap.players_now[snap.players_now.team_code == first.home_code].player_code.iloc[0]
    assert doubled[(code, 3)] == pytest.approx(2 * single[(code, 3)])
    blanked = per_gameweek(predict(fixtures_ahead=blank, **kw))
    assert blanked[blanked.gameweek == 3].empty


def test_new_player_uses_priors_and_is_flagged():
    snap = _snapshot_inputs(synthetic_log())
    newcomer = pd.DataFrame({"player_code": [9999], "team_code": [1], "position": ["MID"], "price": [8.0],
                             "chance": [np.nan]})
    pred = predict(snap.history, pd.concat([snap.players_now, newcomer]), snap.fixtures_ahead, snap.cutoff,
                   "2023-24", snap.teams_in_season, ModelParams())
    row = pred[pred.player_code == 9999].iloc[0]
    assert bool(row.no_history) and row.total > 0 and np.isfinite(row.total)


def test_injury_flag_scales_next_week_and_fades():
    snap = _snapshot_inputs(synthetic_log())
    flagged = snap.players_now.assign(chance=np.where(snap.players_now.player_code == 100, 50.0, np.nan))
    pred = predict(snap.history, flagged, snap.fixtures_ahead, snap.cutoff, "2023-24", snap.teams_in_season,
                   ModelParams(flag_fade=0.5))
    base = predict(snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, "2023-24",
                   snap.teams_in_season, ModelParams(flag_fade=0.5))
    p = pred[pred.player_code == 100].set_index("horizon").p60
    b = base[base.player_code == 100].set_index("horizon").p60
    assert p[0] == pytest.approx(0.5 * b[0]) and p[1] == pytest.approx(0.75 * b[1])


def test_params_round_trip():
    params = ModelParams()
    assert ModelParams.from_dict(params.to_dict()) == params
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_component_model.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# prediction/poisson.py
"""Poisson expectations used by the component model."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

N_MAX = 60  # P(n >= 60) is negligible for every lambda this model produces


def expected_floor_div(lam, k: int) -> np.ndarray:
    """E[floor(N / k)] for N ~ Poisson(lam), element-wise (e.g. -1 per 2 goals conceded, 1 per 3 saves)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    n = np.arange(N_MAX)
    return ((n // k)[:, None] * poisson.pmf(n[:, None], lam[None, :])).sum(axis=0)
```

```python
# prediction/component_model.py
"""Expected FPL points per player, per fixture, from components (design doc section 4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from features.player_rates import RATE_COLUMNS, PlayerParams, player_features, price_band
from features.team_ratings import TeamParams, fit_team_ratings, team_matches

from .poisson import expected_floor_div
from .scoring import ScoringRules, rules_for_season

COMPONENTS = ("appearance", "goals", "assists", "clean_sheet", "conceded", "saves", "bonus", "defensive",
              "discipline")


@dataclass(frozen=True)
class ModelParams:
    team: TeamParams = field(default_factory=TeamParams)
    player: PlayerParams = field(default_factory=PlayerParams)
    flag_fade: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelParams":
        return cls(team=TeamParams(**d["team"]), player=PlayerParams(**d["player"]), flag_fade=d["flag_fade"])


def fixture_components(rows: pd.DataFrame, rules: ScoringRules, components=COMPONENTS) -> pd.DataFrame:
    """Per-component expected points. `rows` needs position, the RATE_COLUMNS, lam_team and lam_opp,
    with p60/psub already scaled for availability."""
    pos, p60, psub = rows["position"], rows["p60"], rows["psub"]
    em = p60 * rows["m60"] + psub * rows["msub"]
    lam_t, lam_o = rows["lam_team"], rows["lam_opp"]
    out = pd.DataFrame(index=rows.index)
    out["appearance"] = 2 * p60 + psub
    out["goals"] = pos.map(rules.goal) * rows["xg_rel"] * lam_t * em / 90
    out["assists"] = rules.assist * rows["xa_rel"] * lam_t * em / 90
    out["clean_sheet"] = pos.map(rules.clean_sheet) * p60 * np.exp(-lam_o)
    out["conceded"] = pos.map(rules.concede_per_two) * p60 * expected_floor_div(lam_o, 2)
    saves_mean = rows["saves_rel"] * lam_o * rows["m60"] / 90
    out["saves"] = (pos == "GKP") * rules.save_per_three * p60 * expected_floor_div(saves_mean, 3)
    out["bonus"] = rows["bonus90"] * em / 90
    out["defensive"] = rules.defensive_contribution * p60 * rows["p_dc"] * (pos != "GKP")
    out["discipline"] = (rules.yellow * rows["yellow90"] + rules.red * rows["red90"]) * em / 90
    for name in COMPONENTS:
        if name not in components:
            out[name] = 0.0
    out["total"] = out[list(COMPONENTS)].sum(axis=1)
    out["exp_minutes"] = em
    return out


def predict(history: pd.DataFrame, players_now: pd.DataFrame, fixtures_ahead: pd.DataFrame,
            cutoff: pd.Timestamp, season: str, teams_in_season: list[int], params: ModelParams,
            components=COMPONENTS) -> pd.DataFrame:
    ratings = fit_team_ratings(team_matches(history), cutoff, season, params.team, teams_in_season)
    feats, priors = player_features(history, cutoff, season, ratings, params.player)

    p = players_now.copy()
    p["band"] = price_band(p["price"])
    p = p.join(feats[RATE_COLUMNS], on="player_code")
    p["no_history"] = p["p60"].isna()
    prior_rows = priors.reindex(pd.MultiIndex.from_arrays([p["position"], p["band"]]))
    for col in RATE_COLUMNS:
        p[col] = p[col].fillna(pd.Series(prior_rows[col].to_numpy(), index=p.index)).fillna(priors[col].mean())

    home = fixtures_ahead.rename(columns={"home_code": "team_code", "away_code": "opponent_code"})
    away = fixtures_ahead.rename(columns={"away_code": "team_code", "home_code": "opponent_code"})
    fx = pd.concat([home.assign(was_home=True), away.assign(was_home=False)], ignore_index=True)
    rows = p.merge(fx, on="team_code")
    if rows.empty:
        return pd.DataFrame(columns=["player_code", "gameweek", "total"])

    rows["horizon"] = rows["gameweek"] - fixtures_ahead["gameweek"].min()
    chance = rows["chance"].fillna(100) / 100
    availability = 1 - (1 - chance) * params.flag_fade ** rows["horizon"]
    rows["p60"] = rows["p60"] * availability
    rows["psub"] = rows["psub"] * availability
    rows["lam_team"] = ratings.expected_goals(rows["team_code"], rows["opponent_code"], rows["was_home"])
    rows["lam_opp"] = ratings.expected_goals(rows["opponent_code"], rows["team_code"],
                                             ~rows["was_home"].astype(bool))
    comps = fixture_components(rows, rules_for_season(season), components)
    keep = ["player_code", "gameweek", "fixture_id", "horizon", "opponent_code", "was_home", "no_history",
            "p60", "psub"]
    return pd.concat([rows[keep], comps], axis=1)


def per_gameweek(pred: pd.DataFrame) -> pd.DataFrame:
    """Sum fixtures within each gameweek (double gameweeks)."""
    return pred.groupby(["player_code", "gameweek"], as_index=False)["total"].sum()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_component_model.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add prediction/poisson.py prediction/component_model.py tests/test_component_model.py
git commit -m "Prediction: component model of expected points per fixture" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Component model in the backtest, and the ablation checkpoint

**Files:**
- Modify: `prediction/backtest.py`
- Test: `tests/test_backtest.py` (add)

**Interfaces:**
- Consumes: `ModelParams`, `predict`, `per_gameweek`, `COMPONENTS` (Task 8); `Snapshot` (Task 5).
- Produces: `make_component_predictor(params: ModelParams, components=COMPONENTS) -> Predictor`; `load_params(path: str | None) -> ModelParams`; backtest CLI accepts `--models component` and `--ablation`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_backtest.py
def test_component_predictor_runs_in_backtest():
    from prediction.backtest import make_component_predictor
    from prediction.component_model import ModelParams
    per_gw, _ = run_backtest(synthetic_log(), ["2023-24"], {"component": make_component_predictor(ModelParams())},
                             horizons=2, decision=False)
    assert set(per_gw.horizon) == {0, 1} and per_gw.mae.notna().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_backtest.py::test_component_predictor_runs_in_backtest -v`
Expected: FAIL (`make_component_predictor` not defined).

- [ ] **Step 3: Implement** (in `prediction/backtest.py`)

```python
import json

from .component_model import COMPONENTS, ModelParams, per_gameweek, predict


def make_component_predictor(params: ModelParams, components=COMPONENTS) -> Predictor:
    def predictor(snap: Snapshot) -> pd.DataFrame:
        pred = predict(snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, snap.season,
                       snap.teams_in_season, params, components)
        return per_gameweek(pred)
    return predictor


def load_params(path: str | None) -> ModelParams:
    if path is None:
        return ModelParams()
    with open(path, encoding="utf-8") as f:
        return ModelParams.from_dict(json.load(f))
```

Replace `build_predictors`:

```python
def build_predictors(names: list[str], params_path: str | None = None,
                     ablation: bool = False) -> dict[str, Predictor]:
    params = load_params(params_path)
    predictors: dict[str, Predictor] = {"naive": naive_predictor, "xp": xp_predictor,
                                        "component": make_component_predictor(params)}
    chosen = {n: predictors[n] for n in names}
    if ablation:  # the component model with one component switched off at a time
        for name in COMPONENTS:
            chosen[f"without_{name}"] = make_component_predictor(
                params, tuple(c for c in COMPONENTS if c != name))
    return chosen
```

In `main`, add `parser.add_argument("--ablation", action="store_true")` and pass `ablation=args.ablation` to `build_predictors`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Checkpoint 1: default parameters vs the benchmarks, and what each component adds**

Run: `python -m prediction.backtest --models naive,xp,component --ablation --every 2 --no-decision`
then: `python -m prediction.backtest --models naive,xp,component`
Expected: summary tables. **Stop and report to the user:** component vs FPL `xP` at horizon 0 (MAE, rho), component vs naive at horizons 1 to 4, decision value, calibration, and for each component how much MAE/rho worsen when it is switched off. A component whose removal does not worsen the metrics is discussed with the user before tuning.

- [ ] **Step 6: Commit**

```bash
git add prediction/backtest.py tests/test_backtest.py
git commit -m "Backtest: component model and per-component ablation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Tuning by coordinate descent

**Files:**
- Create: `prediction/tune.py`
- Test: `tests/test_tune.py`

**Interfaces:**
- Consumes: `run_backtest`, `make_component_predictor`, `xp_predictor` (Task 5/9), `ModelParams`, `TeamParams`, `PlayerParams`.
- Produces: `GRID: dict[str, list[float]]` (keys like `"team.ridge"`, `"player.kappa_xg"`); `objective(component_per_gw, xp_per_gw) -> float` (J); `with_value(params, key, value) -> ModelParams`; `coordinate_descent(evaluate, start, grid, max_passes=3, log=print) -> tuple[ModelParams, float, list[dict]]`; CLI `python -m prediction.tune [--every 2]` writing `data/processed/model_params.json` and `data/processed/tuning_log.csv`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tune.py
import pandas as pd
import pytest

from prediction.component_model import ModelParams
from prediction.tune import coordinate_descent, objective, with_value


def test_objective_is_two_when_level_with_xp():
    same = pd.DataFrame({"horizon": [0, 0], "mae": [2.0, 2.0], "rho": [0.5, 0.5]})
    assert objective(same, same) == pytest.approx(2.0)
    better = same.assign(mae=1.0, rho=0.75)
    assert objective(better, same) == pytest.approx(0.5 + 0.5)


def test_with_value_updates_nested_param():
    p = with_value(ModelParams(), "team.ridge", 7.0)
    assert p.team.ridge == 7.0 and p.player == ModelParams().player


def test_coordinate_descent_finds_the_minimum_of_a_known_bowl():
    # J is minimised at ridge = 4 and kappa_xg = 8
    evaluate = lambda p: (p.team.ridge - 4) ** 2 + (p.player.kappa_xg - 8) ** 2 + 2  # noqa: E731
    grid = {"team.ridge": [1.0, 2.0, 4.0, 6.0], "player.kappa_xg": [2.0, 4.0, 8.0]}
    best, j, history = coordinate_descent(evaluate, ModelParams(), grid, log=lambda *_: None)
    assert best.team.ridge == 4.0 and best.player.kappa_xg == 8.0 and j == pytest.approx(2.0)
    assert len(history) >= 7
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_tune.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# prediction/tune.py
"""Tune the component model's parameters by coordinate descent.

Objective (lower is better; 2.0 = level with FPL's xP):
    J = MAE_model / MAE_xP + (1 - rho_model) / (1 - rho_xP)
on next-gameweek predictions over the tuning seasons only (2025/26 is held out).

    python -m prediction.tune [--every 2]
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import pandas as pd

from ingestion import storage

from .backtest import make_component_predictor, run_backtest, xp_predictor
from .component_model import ModelParams

TUNING_SEASONS = ["2022-23", "2023-24", "2024-25"]
GRID: dict[str, list[float]] = {
    "team.half_life_days": [60, 120, 180, 365],
    "team.ridge": [0.5, 1.0, 2.0, 5.0],
    "team.xg_weight": [0.5, 0.75, 1.0],
    "team.prev_season_fade": [0.25, 0.5, 0.75, 1.0],
    "team.promoted_attack": [-0.3, -0.15, 0.0],
    "team.promoted_defence": [-0.3, -0.15, 0.0],
    "player.half_life_days": [60, 120, 240],
    "player.prev_season_fade": [0.25, 0.5, 0.75],
    "player.kappa_minutes": [1.0, 3.0, 6.0],
    "player.kappa_xg": [2.0, 4.0, 8.0],
    "player.kappa_xa": [2.0, 4.0, 8.0],
    "player.kappa_bonus": [3.0, 6.0, 12.0],
    "player.kappa_saves": [3.0, 6.0, 12.0],
    "player.kappa_dc": [2.0, 5.0, 10.0],
}


def objective(component_per_gw: pd.DataFrame, xp_per_gw: pd.DataFrame) -> float:
    c = component_per_gw[component_per_gw["horizon"] == 0]
    x = xp_per_gw[xp_per_gw["horizon"] == 0]
    return float(c["mae"].mean() / x["mae"].mean() + (1 - c["rho"].mean()) / (1 - x["rho"].mean()))


def with_value(params: ModelParams, key: str, value: float) -> ModelParams:
    group, name = key.split(".")
    return dataclasses.replace(params, **{group: dataclasses.replace(getattr(params, group), **{name: value})})


def coordinate_descent(evaluate, start: ModelParams, grid: dict[str, list[float]], max_passes: int = 3,
                       log=print) -> tuple[ModelParams, float, list[dict]]:
    best, best_j = start, evaluate(start)
    history = [{"pass": 0, "key": "start", "value": None, "J": best_j}]
    log(f"start: J = {best_j:.4f}")
    for n in range(1, max_passes + 1):
        improved = False
        for key, values in grid.items():
            for value in values:
                candidate = with_value(best, key, value)
                if candidate == best:
                    continue
                j = evaluate(candidate)
                history.append({"pass": n, "key": key, "value": value, "J": j})
                if j < best_j - 1e-6:
                    best, best_j, improved = candidate, j, True
                    log(f"pass {n}: {key} = {value} -> J = {j:.4f}")
        if not improved:
            break
    return best, best_j, history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune the component model (tuning seasons only).")
    parser.add_argument("--every", type=int, default=1, help="use every Nth gameweek (faster)")
    args = parser.parse_args(argv)
    log = storage.load_table("archive_match_log")
    tuning_log = log[log["season"].isin(TUNING_SEASONS)]  # 2025/26 never enters tuning
    xp_per_gw, _ = run_backtest(tuning_log, TUNING_SEASONS, {"xp": xp_predictor}, horizons=1,
                                decision=False, every=args.every)

    def evaluate(params: ModelParams) -> float:
        per_gw, _ = run_backtest(tuning_log, TUNING_SEASONS, {"component": make_component_predictor(params)},
                                 horizons=1, decision=False, every=args.every)
        return objective(per_gw, xp_per_gw)

    best, j, history = coordinate_descent(evaluate, ModelParams(), GRID)
    storage.save_json(best.to_dict(), "model_params")
    storage.save_table(pd.DataFrame(history), "tuning_log")
    print(f"Best J = {j:.4f} (2.0 = level with FPL's xP). Saved model_params.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_tune.py -v`
Expected: PASS.

- [ ] **Step 5: Time one evaluation, then tune**

Run: `python -m prediction.tune --every 4` first to measure speed (it prints each improvement). If a full run (`--every 1`) is estimated over two hours, use `--every 2` and confirm the result with a single `--every 1` backtest in Task 11.
Expected: `Best J = ...` and `data/processed/model_params.json` written. **Report the J trajectory and final parameters to the user.**

- [ ] **Step 6: Commit**

```bash
git add prediction/tune.py tests/test_tune.py
git commit -m "Prediction: coordinate-descent tuning on the tuning seasons" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: The single 2025/26 holdout evaluation

**Files:**
- Modify: `README.md` (results section)

**Interfaces:**
- Consumes: `python -m prediction.backtest` with `--params data/processed/model_params.json`.

- [ ] **Step 1: Run the holdout exactly once**

Run: `python -m prediction.backtest --seasons 2025-26 --models naive,xp,component --params data/processed/model_params.json`
Expected: summary and calibration tables for 2025/26. Do not change parameters after this run; if results disappoint, report them honestly and discuss next steps with the user rather than re-tuning on 2025/26.

- [ ] **Step 2: Add a results section to `README.md`** (after "## Optimiser"), filled in with the actual numbers:

```markdown
## Prediction results

Walk-forward backtest on the held-out 2025/26 season (parameters tuned on 2022/23 to 2024/25 only; 2025/26 was
evaluated once). Lower MAE is better; higher rank correlation (rho, within position) is better; decision value is the
average real points per gameweek of the XI and captain the optimiser picks from each model's predictions.

| Model | MAE (next GW) | rho (next GW) | Decision value | MAE (5 GWs ahead) |
|---|---|---|---|---|
| Component model | ... | ... | ... | ... |
| FPL's own xP | ... | ... | ... | n/a |
| Naive (last 5 average) | ... | ... | ... | ... |

Known limitation: the archive holds the final fixture schedule, so predictions more than one week ahead see double
gameweeks slightly earlier than they were announced.
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: held-out 2025/26 prediction results" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Live predictions CLI, optimiser scorer switch, docs

**Files:**
- Create: `prediction/cli.py`
- Modify: `optimisation/cli.py`, `README.md`, local `CLAUDE.md`
- Test: `tests/test_prediction_cli.py`, add to `tests/test_optimisation_cli.py`

**Interfaces:**
- Consumes: `predict`, `per_gameweek`, `ModelParams`, `load_params` pattern; processed tables `match_log`, `players`, `teams`, `fixtures`, `gameweeks`; `transform.season_label`.
- Produces: `prediction.cli.live_inputs(data_dir) -> dict` (keys `history, players_now, fixtures_ahead, cutoff, season, teams_in_season, code_to_id`); `prediction.cli.run(data_dir=storage.DATA_DIR, horizon=5, now=None) -> pd.DataFrame` (per player per fixture with `player_id` added; also saves `predictions.parquet`); `optimisation.cli.run(..., scorer="component")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prediction_cli.py
import pandas as pd

from ingestion import storage
from prediction import cli
from tests.test_backtest import synthetic_log


def write_live_tables(tmp_path):
    log = synthetic_log()
    storage.save_table(log, "match_log", tmp_path)
    players = pd.DataFrame({"id": [1, 2], "code": [100, 203], "team": [1, 2], "position": ["GKP", "MID"],
                            "price": [5.0, 8.0], "chance_of_playing_next_round": [None, 50.0],
                            "can_select": [True, True], "web_name": ["Keeper", "Mid"]})
    teams = pd.DataFrame({"id": [1, 2, 3, 4], "code": [1, 2, 3, 4], "short_name": ["A", "B", "C", "D"]})
    fixtures = pd.DataFrame({"id": [1, 2], "event": [9, 9], "team_h": [1, 3], "team_a": [2, 4],
                             "kickoff_time": pd.to_datetime(["2023-10-07T14:00Z"] * 2, utc=True),
                             "finished": [False, False]})
    gameweeks = pd.DataFrame({"id": [9], "deadline_time": pd.to_datetime(["2023-10-06T17:30Z"], utc=True),
                              "is_next": [True], "is_current": [False], "finished": [False]})
    for name, df in {"players": players, "teams": teams, "fixtures": fixtures, "gameweeks": gameweeks}.items():
        storage.save_table(df, name, tmp_path)


def test_live_predictions_are_saved_with_player_ids(tmp_path):
    write_live_tables(tmp_path)
    pred = cli.run(data_dir=tmp_path, now=pd.Timestamp("2023-10-06T12:00Z"))
    assert set(pred.player_id) == {1, 2} and (pred.gameweek == 9).all()
    assert (tmp_path / "processed" / "predictions.parquet").exists()
    mid = pred[pred.player_id == 2].iloc[0]
    assert 0 < mid.total and mid.p60 <= 0.5 + 1e-9  # 50% flag applied to next week
```

Add to `tests/test_optimisation_cli.py`:

```python
def test_component_scorer_uses_saved_predictions(tmp_path, monkeypatch):
    players = _write_data(tmp_path)
    fake = pd.DataFrame({"player_id": players["id"], "gameweek": 9, "total": 2.0})
    monkeypatch.setattr("prediction.cli.run", lambda data_dir, **kw: fake)
    sol, context = cli.run(data_dir=tmp_path, scorer="component")
    assert len(sol.squad) == 15 and context["scores"].eq(2.0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_prediction_cli.py tests/test_optimisation_cli.py -v`
Expected: FAIL (`prediction.cli` not found; `run()` has no `scorer`).

- [ ] **Step 3: Implement `prediction/cli.py`**

```python
# prediction/cli.py
"""Live expected points for the next five gameweeks.

    python -m prediction.cli [--position MID] [--top 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ingestion import storage, transform

from .component_model import ModelParams, predict

PARAMS_FILE = "model_params"


def load_live_params(data_dir: Path) -> ModelParams:
    stored = storage.load_json_or_none(PARAMS_FILE, data_dir)
    if stored is None:
        print("  model_params.json not found: using default parameters (run `python -m prediction.tune`)")
        return ModelParams()
    return ModelParams.from_dict(stored)


def live_inputs(data_dir: Path, horizon: int = 5, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    players = storage.load_table("players", data_dir)
    teams = storage.load_table("teams", data_dir)
    fixtures = storage.load_table("fixtures", data_dir)
    gameweeks = storage.load_table("gameweeks", data_dir)
    code_of = teams.set_index("id")["code"]
    next_gw = int(gameweeks.loc[gameweeks["is_next"], "id"].iloc[0])
    ahead = fixtures[(fixtures["event"] >= next_gw) & (fixtures["event"] < next_gw + horizon)]
    selectable = players[players["can_select"].fillna(True).astype(bool)]
    return {
        "history": storage.load_table("match_log", data_dir),
        "players_now": pd.DataFrame({
            "player_code": selectable["code"], "team_code": selectable["team"].map(code_of),
            "position": selectable["position"], "price": selectable["price"],
            "chance": selectable["chance_of_playing_next_round"]}).reset_index(drop=True),
        "fixtures_ahead": pd.DataFrame({
            "gameweek": ahead["event"].astype(int), "fixture_id": ahead["id"], "kickoff": ahead["kickoff_time"],
            "home_code": ahead["team_h"].map(code_of), "away_code": ahead["team_a"].map(code_of)}),
        "cutoff": now,
        "season": transform.season_label(gameweeks),
        "teams_in_season": sorted(code_of.tolist()),
        "code_to_id": players.set_index("code")["id"],
    }


def run(data_dir: Path = storage.DATA_DIR, horizon: int = 5, now: pd.Timestamp | None = None) -> pd.DataFrame:
    x = live_inputs(data_dir, horizon, now)
    pred = predict(x["history"], x["players_now"], x["fixtures_ahead"], x["cutoff"], x["season"],
                   x["teams_in_season"], load_live_params(data_dir))
    pred.insert(0, "player_id", pred["player_code"].map(x["code_to_id"]))
    storage.save_table(pred, "predictions", data_dir)
    return pred


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Expected points for the next five gameweeks.")
    parser.add_argument("--position", choices=["GKP", "DEF", "MID", "FWD"])
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args(argv)
    pred = run()
    players = storage.load_table("players").set_index("id")
    table = pred.groupby(["player_id", "gameweek"])["total"].sum().unstack("gameweek").round(2)
    table["5 GW total"] = table.sum(axis=1)
    info = players.loc[table.index, ["web_name", "team_short", "position", "price"]]
    first = pred.groupby("player_id").first()
    table = info.join(table).join(first[["p60", "no_history"]])
    if args.position:
        table = table[table["position"] == args.position]
    print(table.sort_values("5 GW total", ascending=False).head(args.top).to_string())


if __name__ == "__main__":
    main()
```

Note: `season_label` requires `deadline_time` sorted by `id`; the test's single-row gameweeks table yields `"2023-24"`.

- [ ] **Step 4: Switch the optimiser's default score** (in `optimisation/cli.py`)

Add `scorer: str = "component"` to `run(...)`; replace `scores = score_players(players, ep_weight)` with:

```python
    if scorer == "component":
        from prediction import cli as prediction_cli
        pred = prediction_cli.run(data_dir=data_dir)
        next_gw = pred["gameweek"].min()
        scores = pred[pred["gameweek"] == next_gw].groupby("player_id")["total"].sum()
    else:
        scores = score_players(players, ep_weight)
```

and add `parser.add_argument("--scorer", choices=["component", "placeholder"], default="component")`, passing `scorer=args.scorer` to `run`. In `format_solution`, the `score` column already prints `context["scores"]`, which is now expected points.

The Task 1 tests have no match log, so they must keep the Stage 2 score: in `tests/test_optimisation_cli.py`, change `cli.run(data_dir=tmp_path)` to `cli.run(data_dir=tmp_path, scorer="placeholder")` and `cli.run(entry_id=42, data_dir=tmp_path)` to `cli.run(entry_id=42, data_dir=tmp_path, scorer="placeholder")`.

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q` then `python -m pytest -m live -q`
Expected: all pass.

- [ ] **Step 6: Live check on the user's team**

Run: `python -m prediction.cli --top 15` then `python -m optimisation.cli --entry 8298351`
Expected: a five-week predictions table with components and `p60`; a legal transfer plan scored by the component model. Compare with `--scorer placeholder` and report differences.

- [ ] **Step 7: Docs**

- `README.md`: Quick start gains `python -m ingestion.archive` (once), `python -m prediction.cli`, `python -m prediction.backtest`, `python -m prediction.tune`; the "Stage 2 scores players with ..." paragraph is replaced by a short description of the component model linking `docs/stage-3a-prediction-design.md`; Status marks Stage 3a done.
- Local `CLAUDE.md`: Section 3 adds `scipy` and the archive data source; Section 5 marks Stage 3a; Section 7 status summarises 3a with the holdout numbers; Section 8 changelog entry.

- [ ] **Step 8: Commit**

```bash
git add prediction/cli.py optimisation/cli.py tests README.md
git commit -m "Live component predictions; optimiser uses them by default" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Self-review against the spec

| Spec requirement | Task |
|---|---|
| 3a scope: CLI test, prediction, backtest | 1, 3 to 12 |
| Archive 2022/23 to 2025/26, stable codes, licence note | 3 |
| Current season via `event/{gw}/live` | 4 |
| Match log schema | 3, 4 |
| Team ratings (Poisson, xG target alpha, decay, ridge, promoted prior) | 6 |
| Minutes model with shrinkage, typical minutes, flag fade | 7, 8 |
| Attacking rates in team context, position and price-band priors | 7 |
| Components incl. per-season scoring, Poisson helpers | 2, 8 |
| Double and blank gameweeks | 8 |
| Walk-forward replay, cutoff, eligibility, warm-up | 5 |
| Metrics: MAE, RMSE, rho, decision value (real optimiser), calibration | 5 |
| Tuning objective J, coordinate descent, 2025/26 locked | 10, 11 |
| Leakage safeguards (per-row team/price, pre-cutoff priors, holdout, xP never an input) | 3, 5, 6, 7, 10 |
| Live CLI, `predictions.parquet`, `model_params.json`, defaults with warning, `no_history` flag | 8, 12 |
| Optimiser `--scorer component` default, placeholder kept, ILP unchanged | 12 |
| scipy dependency | 6 |
| Error handling: archive failure message, missing params, non-convergence, no-history | 3 (raises with URL), 12, 6, 8 |
| Results table in README | 11 |
