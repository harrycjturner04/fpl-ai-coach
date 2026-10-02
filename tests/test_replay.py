import numpy as np
import pandas as pd
import pytest

from evaluation.replay import HOLD, SINGLE_WEEK, replay_season, score_table
from optimisation import settings as settings_module
from optimisation.settings import PlanSettings, load_settings
from ingestion.transform import MATCH_COLUMNS
from prediction.backtest import naive_predictor

FAST = PlanSettings(horizon=2, leftover_values=(1.5, 1.5, 1.5, 1.5))
PAIRINGS = [[(1, 2), (3, 4), (5, 6)], [(1, 3), (2, 5), (4, 6)], [(1, 4), (2, 6), (3, 5)]]


def league_log(n_gw=7, seed=0):
    """Two seasons, 6 clubs x 6 players (GKP, 2 DEF, 2 MID, FWD) so a legal 15 exists under the
    3-per-club cap and there are spare players to transfer. Extends test_backtest.synthetic_log
    (4 clubs only allow 12 players under the cap); club 1 players rise in price from gameweek 5."""
    rng = np.random.default_rng(seed)
    positions = ["GKP", "DEF", "DEF", "MID", "MID", "FWD"]
    rows = []
    for s_i, season in enumerate(["2022-23", "2023-24"]):
        for gw in range(1, n_gw + 1):
            kickoff = pd.Timestamp("2022-08-06", tz="UTC") + pd.Timedelta(days=365 * s_i + 7 * gw)
            for f_i, (h, a) in enumerate(PAIRINGS[gw % 3]):
                for team, opp, home in ((h, a, True), (a, h, False)):
                    for k, pos in enumerate(positions):
                        minutes = 90 if k < 5 else int(rng.choice([0, 20, 90]))
                        rows.append({c: 0 for c in MATCH_COLUMNS} | {
                            "season": season, "gameweek": gw, "fixture_id": gw * 10 + f_i, "kickoff": kickoff,
                            "player_code": team * 100 + k, "team_code": team, "opponent_code": opp,
                            "was_home": home, "position": pos, "minutes": minutes,
                            "xg": 0.3 if pos in ("MID", "FWD") and minutes else 0.0,
                            "points": int(rng.integers(0, 10)) if minutes else 0,
                            "price": 4.5 + 0.5 * k + (0.3 if team == 1 and gw >= 5 else 0.0),
                            "xp": float(rng.uniform(1, 5)),
                        })
    return pd.DataFrame(rows)


LOG = league_log()


def run(settings=FAST, log=LOG, **kw):
    return replay_season(log, "2023-24", naive_predictor, settings, **kw)


def test_one_row_per_gameweek_and_deterministic():
    a = run()
    assert list(a.columns) == ["season", "gameweek", "points", "hits", "transfers", "free_transfers", "bank", "predicted"]
    assert list(a.gameweek) == list(range(1, 8))
    pd.testing.assert_frame_equal(a, run())


def test_hold_never_transfers_after_first_week():
    r = run(HOLD)
    assert (r.transfers == 0).all() and (r.hits == 0).all()


def test_single_week_runs():
    assert len(run(SINGLE_WEEK)) == 7


def test_bookkeeping():
    r = run(PlanSettings(horizon=2, hit_margin=-10.0)).reset_index(drop=True)  # negative margin: hits are attractive
    assert r.free_transfers.iloc[0] == 0 and r.transfers.iloc[0] == 0
    for _, row in r.iterrows():
        assert row.hits == max(0, row.transfers - row.free_transfers)
    for prev, nxt in zip(r.itertuples(), r.iloc[1:].itertuples()):
        assert nxt.free_transfers == min(5, max(prev.free_transfers - prev.transfers, 0) + 1)


def test_predictor_never_sees_the_future():
    calls = []

    def checked(snap):
        assert snap.history["kickoff"].max() < snap.cutoff
        calls.append(snap.gameweek)
        return naive_predictor(snap)

    replay_season(LOG, "2023-24", checked, FAST)
    assert calls == list(range(1, 8))


def test_cache_is_filled_and_used():
    cache = {}
    run(cache=cache)
    assert set(cache) == {("2023-24", g) for g in range(1, 8)}

    def boom(snap):
        raise AssertionError("predictor called despite a full cache")

    replay_season(LOG, "2023-24", boom, FAST, cache=cache)


def test_player_who_left_the_league_scores_zero_and_does_not_crash():
    states = run(HOLD, keep_states=True)["state"]
    gone = sorted(states.iloc[1]["current_squad"])[0]
    log = LOG[~((LOG.player_code == gone) & (LOG.season == "2023-24") & (LOG.gameweek > 3))]
    r = run(HOLD, log=log, keep_states=True)
    assert len(r) == 7
    assert gone in r.iloc[5]["state"]["current_squad"]  # held with a last known price
    assert gone in set(r.iloc[5]["state"]["players"]["id"])


def test_keep_states_allows_a_resolve():
    from optimisation.model import solve_plan
    from prediction.backtest import STANDARD_RULES

    st = run(keep_states=True).iloc[3]["state"]
    plan = solve_plan(st["players"], st["scores"], STANDARD_RULES, **{k: v for k, v in st.items() if k not in ("players", "scores")} | {"free_transfers": 2})
    assert plan.weeks[0].free_transfers_next is not None


def test_score_table_summed_discounts():
    pred = pd.DataFrame({"player_code": [1, 1, 2, 2], "gameweek": [3, 4, 3, 4], "total": [2.0, 4.0, 1.0, 3.0]})
    t = score_table(pred, 3, PlanSettings(horizon=2, discount=0.5, summed=True))
    assert list(t.columns) == [3]
    assert t.loc[1, 3] == pytest.approx(2.0 + 0.5 * 4.0, abs=1e-4)
    assert t.loc[2, 3] == pytest.approx(1.0 + 0.5 * 3.0, abs=1e-4)


def test_score_table_only_existing_gameweeks_and_fills_zero():
    pred = pd.DataFrame({"player_code": [1, 2], "gameweek": [3, 4], "total": [2.0, 4.0]})
    t = score_table(pred, 3, PlanSettings(horizon=5))
    assert list(t.columns) == [3, 4]
    assert t.loc[1, 4] == 0 and t.loc[2, 3] == 0


def test_settings_roundtrip(tmp_path):
    s = PlanSettings(horizon=3, discount=0.9, leftover_values=(1.0, 0.5), hold=True)
    assert PlanSettings.from_dict(s.to_dict()) == s
    assert isinstance(s.to_dict()["leftover_values"], list)


def test_load_settings_empty_dir(tmp_path):
    shipped = settings_module.SHIPPED_PARAMS_PATH
    if shipped.exists():
        import json
        assert load_settings(tmp_path) == PlanSettings.from_dict(json.loads(shipped.read_text()))
    else:
        assert load_settings(tmp_path) == PlanSettings()
