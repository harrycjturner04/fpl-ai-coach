import numpy as np
import pandas as pd
import pytest

import evaluation.tune as tune
from evaluation.tune import (config_key, evaluate, leave_one_season_out, load_results, points_bootstrap, replayed_gameweeks,
                             run_replay, total_points, with_setting)
from optimisation.settings import PlanSettings
from prediction.backtest import naive_predictor, snapshot
from prediction.backtest import WARM_UP_GAMEWEEKS
from tests.test_replay import FAST, league_log

SEASONS = ["s1", "s2", "s3", "s4"]


def table(rows):
    return pd.DataFrame([dict(config=c, season=s, points=p) for c, ps in rows.items() for s, p in zip(SEASONS, ps)])


def test_leave_one_season_out_picks_best_on_the_other_seasons():
    res = table({"A": [10, 10, 10, 1], "B": [9, 9, 9, 50]})
    out = leave_one_season_out(res)
    assert list(out["season"]) == SEASONS + ["total"]
    # held out s1-s3: A (21 + ... = 21) vs B (9+9+50 = 68) -> B; held out s4: A (30) vs B (27) -> A
    assert list(out["config"][:4]) == ["B", "B", "B", "A"]
    assert list(out["points"]) == pytest.approx([9, 9, 9, 1, 28], abs=1e-4)


def test_leave_one_season_out_tie_picks_the_first_config():
    res = table({"A": [5, 5, 5, 5], "B": [5, 5, 5, 5]})
    assert set(leave_one_season_out(res)["config"][:4]) == {"A"}


def frames(diff, n=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame(dict(season="s", gameweek=range(1, n + 1), points=rng.normal(50, 10, n)))
    return base.assign(points=base.points + diff + rng.normal(0, noise, n)), base


def test_bootstrap_excludes_zero_for_constant_difference():
    a, b = frames(2.0)
    r = points_bootstrap(a, b)
    assert r["difference"] == pytest.approx(2.0, abs=1e-4)
    assert r["low"] > 0 and r["gameweeks"] == 60


def test_bootstrap_includes_zero_for_noise():
    a, b = frames(0.0, noise=10.0)
    r = points_bootstrap(a, b)
    assert r["low"] < 0 < r["high"]


def test_bootstrap_counts_shared_pairs_and_raises_when_none():
    a, b = frames(0.0)
    assert points_bootstrap(a, b.iloc[:10])["gameweeks"] == 10
    with pytest.raises(ValueError):
        points_bootstrap(a, b.assign(season="other"))


def test_with_setting_replaces_one_field():
    s = PlanSettings()
    assert with_setting(s, "horizon", 3, (1.0,) * 4) == PlanSettings(horizon=3)


def test_leftover_scale_scales_the_base_table_only():
    base = (3.0, 1.0, 0.5, 0.0)
    s = PlanSettings(discount=0.9)
    out = with_setting(s, "leftover_scale", 0.5, base)
    assert out.leftover_values == pytest.approx((1.5, 0.5, 0.25, 0.0), abs=1e-4)
    assert out.discount == 0.9 and out.horizon == s.horizon


def test_config_key_is_stable_and_distinguishes_settings():
    assert config_key(PlanSettings()) == config_key(PlanSettings())
    assert config_key(PlanSettings()) != config_key(PlanSettings(horizon=3))
    assert config_key(PlanSettings()) != config_key(PlanSettings(leftover_values=(0.0,) * 4))


@pytest.fixture
def worker_globals(monkeypatch):
    log = league_log()
    cache = {("2023-24", gw): naive_predictor(snapshot(log, "2023-24", gw, 5)) for gw in range(1, 8)}
    monkeypatch.setattr(tune, "_LOG", log)
    monkeypatch.setattr(tune, "_CACHE", cache)
    return cache


def test_run_replay_reads_everything_from_the_cache(worker_globals):
    frame = run_replay((FAST, "2023-24", False, None))
    assert list(frame.gameweek) == list(range(1, 8))


def test_run_replay_cache_miss_raises(worker_globals):
    del worker_globals[("2023-24", 3)]
    with pytest.raises(RuntimeError):
        run_replay((FAST, "2023-24", False, None))


def test_run_replay_limit_truncates_the_season(worker_globals):
    assert list(run_replay((FAST, "2023-24", False, 3)).gameweek) == [1, 2, 3]


def test_replayed_gameweeks_skips_warm_up_only_in_the_first_season():
    log = league_log(n_gw=7)
    assert replayed_gameweeks(log, "2022-23") == [g for g in range(1, 8) if g > WARM_UP_GAMEWEEKS]
    assert replayed_gameweeks(log, "2023-24") == list(range(1, 8))
    assert replayed_gameweeks(log, "2022-23")[:2] == [WARM_UP_GAMEWEEKS + 1, WARM_UP_GAMEWEEKS + 2]


def test_limit_keeps_earlier_seasons_intact(worker_globals, monkeypatch):
    seen = {}

    def spy(log, season, predictor, settings, **kw):
        seen["log"] = log
        return pd.DataFrame()

    monkeypatch.setattr(tune, "replay_season", spy)
    run_replay((FAST, "2023-24", False, 3))
    log = seen["log"]
    assert (log[log["season"] == "2022-23"].shape[0]) == (tune._LOG["season"] == "2022-23").sum()
    assert log[log["season"] == "2023-24"]["gameweek"].max() == 3


def test_total_points_sums_over_frames():
    frames = [pd.DataFrame(dict(points=[1.5, 2.0])), pd.DataFrame(dict(points=[3.0]))]
    assert total_points(frames) == pytest.approx(6.5, abs=1e-4)


class FakePool:
    map = staticmethod(map)


def test_evaluate_saves_and_reload_restores_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "run_replay", lambda job: pd.DataFrame(
        dict(season=job[1], gameweek=[1, 2], points=[job[0].horizon * 1.0, 2.0], hits=0)))
    settings = [PlanSettings(horizon=h) for h in (5, 3, 4)]
    results = {}
    evaluate(settings, FakePool(), ["s1", "s2"], results, None, data_dir=tmp_path, name="res")
    loaded = load_results("res", tmp_path)
    assert list(loaded) == list(results) == [config_key(s) for s in settings]
    for key in results:
        for season, frame in results[key].items():
            assert "state" not in loaded[key][season].columns
            pd.testing.assert_frame_equal(loaded[key][season], frame)
