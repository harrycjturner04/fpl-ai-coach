import numpy as np
import pandas as pd
import pytest

import evaluation.replay as replay_module
from evaluation.replay import HOLD, SINGLE_WEEK, SUMMED, replay_season, score_table
from optimisation import settings as settings_module
from optimisation.settings import PlanSettings, load_settings
from ingestion.manager_state import selling_price
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


PRICES = LOG[LOG.season == "2023-24"].set_index(["gameweek", "player_code"])["price"]


def check_bank_flow(r):
    """Recompute each week's bank from the squads entering the weeks (kept states) and the log's prices;
    returns the sales as (week index, player, credited price)."""
    squads = [set(s["current_squad"]) for s in r["state"].iloc[1:]]   # squad entering week 1, 2, ...
    purchase = {p: PRICES[1, p] for p in squads[0]}                    # original squad bought in gameweek 1
    sales = []
    for k in range(1, len(squads)):                                     # week k, gameweek k + 1
        before, after, gw = squads[k - 1], squads[k], k + 1
        credit = {p: selling_price(purchase[p], PRICES[gw, p]) for p in before - after}
        spend = sum(PRICES[gw, p] for p in after - before)
        assert r.bank.iloc[k] == pytest.approx(r.bank.iloc[k - 1] + sum(credit.values()) - spend, abs=1e-4)
        purchase = {p: purchase.get(p, PRICES[gw, p]) for p in after}
        sales += [(k, p, c) for p, c in credit.items()]
    return sales


def test_bookkeeping():
    r = run(PlanSettings(horizon=2, hit_margin=-10.0), keep_states=True).reset_index(drop=True)  # hits are attractive
    assert r.transfers.sum() > 0 and r.hits.sum() > 0
    assert r.free_transfers.iloc[0] == 0 and r.transfers.iloc[0] == 0
    for _, row in r.iterrows():
        assert row.hits == max(0, row.transfers - row.free_transfers)
    for prev, nxt in zip(r.itertuples(), r.iloc[1:].itertuples()):
        assert nxt.free_transfers == min(5, max(prev.free_transfers - prev.transfers, 0) + 1)
    check_bank_flow(r)


def test_sale_after_a_price_rise_credits_half_the_rise():
    def club_one_early(snap):  # club 1 looks great before the rise, worthless from gameweek 5
        pred = naive_predictor(snap)
        club1 = (pred.player_code // 100) == 1
        pred.loc[club1, "total"] = 20.0 if snap.gameweek < 5 else 0.0
        return pred

    r = replay_season(LOG, "2023-24", club_one_early, PlanSettings(horizon=1, hit_margin=-10.0), keep_states=True)
    sales = check_bank_flow(r)
    club1_sales = [(p, c) for k, p, c in sales if k == 4 and p // 100 == 1]  # week index 4 is gameweek 5
    assert club1_sales
    for p, credit in club1_sales:
        bought = PRICES[1, p]    # all of club 1 was bought in gameweek 1 at the pre-rise price
        assert PRICES[5, p] == pytest.approx(bought + 0.3, abs=1e-4)
        assert credit == pytest.approx(bought + 0.1, abs=1e-4)   # half of a 0.3 rise, rounded down


def test_player_bought_in_a_price_change_week_is_sold_at_that_purchase_price():
    def club_one_only_in_gw5(snap):  # club 1 is wanted exactly in gameweek 5, the week its prices rise
        pred = naive_predictor(snap)
        club1 = (pred.player_code // 100) == 1
        pred.loc[club1, "total"] = 20.0 if snap.gameweek == 5 else -5.0
        return pred

    r = replay_season(LOG, "2023-24", club_one_only_in_gw5, PlanSettings(horizon=1, hit_margin=-10.0), keep_states=True)
    sales = check_bank_flow(r)
    squads = [set(s["current_squad"]) for s in r["state"].iloc[1:]]   # squads[k] is held after gameweek k + 1
    bought_in_gw5 = {p for p in squads[4] - squads[3] if p // 100 == 1}
    later_sales = [(p, c) for k, p, c in sales if k >= 5 and p in bought_in_gw5]
    assert later_sales
    for p, credit in later_sales:
        assert PRICES[5, p] == pytest.approx(PRICES[4, p] + 0.3, abs=1e-4)      # the price changed in the week of purchase
        assert credit == pytest.approx(PRICES[5, p], abs=1e-4)                   # no rise since purchase: full price back
        assert credit != pytest.approx(selling_price(PRICES[4, p], PRICES[5, p]), abs=1e-4)  # last week's price would differ


def test_predictor_never_sees_the_future():
    calls = []

    def checked(snap):
        assert snap.history["kickoff"].max() < snap.cutoff
        assert list(snap.players_now.columns) == ["player_code", "team_code", "position", "price", "chance"]
        calls.append(snap.gameweek)
        return naive_predictor(snap)

    replay_season(LOG, "2023-24", checked, FAST)
    assert calls == list(range(1, 8))


def test_changing_the_future_cannot_change_the_past():
    first_changed = 5
    rng = np.random.default_rng(1)
    future = (LOG.season == "2023-24") & (LOG.gameweek >= first_changed)
    scrambled = LOG.copy()
    n = int(future.sum())
    outcomes = ["minutes", "starts", "goals", "assists", "xg", "xa", "xgc", "clean_sheets", "goals_conceded",
                "own_goals", "penalties_saved", "penalties_missed", "saves", "bonus", "defensive_contribution",
                "yellow_cards", "red_cards", "points", "xp"]   # everything about a match except who, where and price
    for col in outcomes:   # most are all zero in the synthetic league: give them values so a leak would show
        scrambled[col] = scrambled[col].astype(float)
        scrambled.loc[future, col] = rng.uniform(0, 8, n)
    scrambled.loc[future, "minutes"] = rng.choice([0, 45, 90], n)
    scrambled.loc[future, "points"] = rng.integers(-2, 15, n)
    a, b = run(), run(log=scrambled)
    assert not a.equals(b)   # the scramble does matter from the changed gameweek on
    pd.testing.assert_frame_equal(a[a.gameweek < first_changed].reset_index(drop=True),
                                  b[b.gameweek < first_changed].reset_index(drop=True))
    # the decision in the changed gameweek itself cannot depend on that gameweek's outcomes either
    own = lambda df: df[df.gameweek == first_changed].drop(columns="points").reset_index(drop=True)
    pd.testing.assert_frame_equal(own(a), own(b))


def test_cache_is_filled_and_used():
    cache = {}
    run(cache=cache)
    assert set(cache) == {("2023-24", g) for g in range(1, 8)}

    def boom(snap):
        raise AssertionError("predictor called despite a full cache")

    replay_season(LOG, "2023-24", boom, FAST, cache=cache)


def test_player_who_left_the_league_scores_zero_and_does_not_crash(monkeypatch):
    states = run(HOLD, keep_states=True)["state"]
    gone = sorted(states.iloc[1]["current_squad"])[0]
    log = LOG[~((LOG.player_code == gone) & (LOG.season == "2023-24") & (LOG.gameweek > 3))]
    seen = []
    real = replay_module.score_gameweek
    monkeypatch.setattr(replay_module, "score_gameweek", lambda st, bn, c, v, minutes, points, *a, **k: (
        seen.append((set(st) | set(bn), minutes.get(gone, 0), points.get(gone, 0))), real(st, bn, c, v, minutes, points, *a, **k))[1])
    r = run(HOLD, log=log, keep_states=True)
    assert len(r) == 7
    for squad, minutes, points in seen[3:]:   # gameweeks 4 to 7
        assert gone in squad and minutes == 0 and points == 0
    state = r.iloc[5]["state"]
    last_known = log[(log.player_code == gone) & (log.season == "2023-24") & (log.gameweek == 3)].iloc[0]
    row = state["players"].set_index("id").loc[gone]
    assert (row.price, row.team, row.position) == (last_known.price, last_known.team_code, last_known.position)
    assert state["current_squad"][gone] == pytest.approx(selling_price(PRICES[1, gone], last_known.price), abs=1e-4)


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


def test_summed_benchmark_is_one_column_of_the_discounted_five_week_sum():
    pred = pd.DataFrame([dict(player_code=1, gameweek=g, total=float(10 ** (g - 3))) for g in range(3, 8)])
    t = score_table(pred, 3, SUMMED)
    assert SUMMED.horizon == 5 and t.shape == (1, 1)
    assert t.iloc[0, 0] == pytest.approx(sum(0.85 ** h * 10.0 ** h for h in range(5)), abs=1e-4)


def test_score_table_only_existing_gameweeks_and_fills_zero():
    pred = pd.DataFrame({"player_code": [1, 2], "gameweek": [3, 4], "total": [2.0, 4.0]})
    t = score_table(pred, 3, PlanSettings(horizon=5))
    assert list(t.columns) == [3, 4]
    assert t.loc[1, 4] == 0 and t.loc[2, 3] == 0


def test_settings_roundtrip():
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
