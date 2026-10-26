import json
import numpy as np, pandas as pd
import pytest
from features.team_ratings import TeamParams
from prediction.ml_data import deadline_rows
from prediction.walkforward import _log_loss, minutes_report, retrain_blocks, run_minutes, training_rows
from tests.test_backtest import synthetic_log


def _table():
    log = synthetic_log(n_gw=8)
    table = pd.concat([deadline_rows(log, s, g, TeamParams()) for s, g in
                      [("2022-23", 6), ("2022-23", 7), ("2023-24", 1), ("2023-24", 2), ("2023-24", 5)]],
                     ignore_index=True)
    return table.assign(t_minutes=table["t_minutes"].where(table.index % 7 != 0, 20))   # all three classes present


def test_retrain_blocks_cover_every_deadline_once():
    table = pd.DataFrame({"season": "s", "deadline_gw": range(1, 11)})
    blocks = retrain_blocks(table, "s")
    assert blocks[0] == (1, [1, 2, 3, 4]) and blocks[-1] == (9, [9, 10])
    assert sorted(g for _, gs in blocks for g in gs) == list(range(1, 11))


def test_training_rows_for_a_block_are_all_before_its_cutoff():
    table = _table()
    cutoff = table.loc[(table["season"] == "2023-24") & (table["deadline_gw"] == 5), "cutoff"].iloc[0]
    rows = training_rows(table, "2023-24", 5)
    assert (rows["kickoff"] < cutoff).all() and len(rows) > 0


def test_run_minutes_returns_probabilities_for_every_test_deadline_and_reports(tmp_path):
    table = _table()
    oos = run_minutes(table, "lgbm", seasons=("2023-24",), params={"n_estimators": 20}, workers=1, data_dir=tmp_path)
    assert set(oos["deadline_gw"]) == {1, 2, 5}
    assert np.allclose(oos[["p0", "p1", "p2"]].sum(axis=1), 1)
    comp = oos.assign(p60=oos["p2"], psub=oos["p1"])
    ll, cal = minutes_report({"lgbm": oos}, table, comp)
    assert set(ll["model"]) == {"lgbm", "component"} and (ll["log_loss"] > 0).all() and len(cal)


def test_training_rows_exclude_a_later_season():
    table = _table()
    assert (training_rows(table, "2022-23", 7)["season"] == "2022-23").all()


def test_log_loss_known_numbers_and_component_p0_clipping():
    p = np.array([[0.5, 0.25, 0.25], [0.1, 0.1, 0.8]])
    assert np.isclose(_log_loss(p, np.array([0, 2])), (np.log(2) + -np.log(0.8)) / 2)
    comp = pd.DataFrame({"p60": [0.7], "psub": [0.4]})       # p0 = 1 - 0.7 - 0.4 = -0.1, clipped to 1e-6 then renormalised
    p0 = (1 - comp["p60"] - comp["psub"]).to_numpy()
    q = np.c_[p0, comp["psub"], comp["p60"]]
    assert np.isclose(_log_loss(q, np.array([0])), -np.log(1e-6 / (1e-6 + 0.4 + 0.7)))


def test_ml_minutes_predictor_uses_p2_as_p60_p1_as_psub_and_keeps_others():
    from prediction.backtest import snapshot
    from prediction.component_model import ModelParams, per_gameweek, predict
    from prediction.walkforward import make_ml_minutes_predictor
    snap = snapshot(synthetic_log(), "2023-24", 3)
    args = (snap.history, snap.players_now, snap.fixtures_ahead, snap.cutoff, "2023-24", snap.teams_in_season,
            ModelParams())
    base = predict(*args)
    one = base[base.player_code == 100]
    oos = one[["player_code", "fixture_id"]].assign(season="2023-24", deadline_gw=3, p0=1.0, p1=0.0, p2=0.0)
    out = make_ml_minutes_predictor(ModelParams(), oos)(snap).set_index(["player_code", "gameweek"]).total
    ref = per_gameweek(base).set_index(["player_code", "gameweek"]).total
    assert (out[100] < ref[100]).all()                       # p60 forced to 0 lowers the player
    rest = out.drop(100, level=0)
    assert rest.to_numpy() == pytest.approx(ref.drop(100, level=0).reindex(rest.index).to_numpy())
    # no oos rows for the deadline: unchanged
    none = make_ml_minutes_predictor(ModelParams(), oos.assign(deadline_gw=99))(snap)
    assert none.set_index(["player_code", "gameweek"]).total.to_numpy() == pytest.approx(ref.to_numpy())


def test_a_later_deadline_oos_row_does_not_change_an_earlier_snapshot():
    from prediction.backtest import snapshot
    from prediction.component_model import ModelParams
    from prediction.walkforward import make_ml_minutes_predictor
    snap = snapshot(synthetic_log(), "2023-24", 3)
    rows = snap.fixtures_ahead
    fx = rows.iloc[0].fixture_id
    oos = pd.DataFrame(dict(season="2023-24", deadline_gw=3, player_code=100, fixture_id=fx, p0=0.0, p1=0.0, p2=1.0),
                       index=[0])
    later = oos.assign(deadline_gw=4, p0=1.0, p2=0.0)
    a = make_ml_minutes_predictor(ModelParams(), oos)(snap)
    b = make_ml_minutes_predictor(ModelParams(), pd.concat([oos, later]))(snap)
    assert a.total.to_numpy() == pytest.approx(b.total.to_numpy())


def test_training_table_keys_are_unique():
    table = _table()
    assert not table.duplicated(["season", "deadline_gw", "player_code", "fixture_id"]).any()


# ---- live use ----
from prediction import walkforward as wf


class _Fake:
    def __init__(self, v=1):
        self.v = v


def _live_files(tmp_path, through, n_gw=8):
    from ingestion import storage
    log = synthetic_log(n_gw=n_gw)
    log = log[(log["season"] == "2023-24") & (log["gameweek"] <= through)]
    first = log.groupby("gameweek")["kickoff"].min()
    gws = pd.DataFrame({"id": range(1, n_gw + 1),
                        "deadline_time": [pd.Timestamp("2023-08-12", tz="UTC") + pd.Timedelta(days=7 * g) for g in range(1, n_gw + 1)],
                        "finished": [g <= through for g in range(1, n_gw + 1)]})
    storage.save_table(log, "match_log", tmp_path)
    storage.save_table(gws, "gameweeks", tmp_path)
    return first


def test_train_live_trains_when_nothing_is_saved_then_only_every_retrain_every_gameweeks(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(wf, "_fit_models", lambda *a: calls.append(1) or {k: _Fake(3) for k in wf.ALL_KINDS})
    first = _live_files(tmp_path, through=4)
    folder = wf.train_live(tmp_path, pd.Timestamp("2023-10-01", tz="UTC"))
    assert calls == [1] and folder.parent == tmp_path / "models" and folder.name == str(first[4].date())
    assert (folder / "meta.json").exists() and wf._load_models(tmp_path, "2023-24")["minutes"].v == 3
    _live_files(tmp_path, through=4 + wf.RETRAIN_EVERY - 1)         # one gameweek short of the cadence
    assert wf.train_live(tmp_path, pd.Timestamp("2023-10-01", tz="UTC")) == folder and calls == [1]
    first = _live_files(tmp_path, through=4 + wf.RETRAIN_EVERY)
    new = wf.train_live(tmp_path, pd.Timestamp("2023-12-01", tz="UTC"))
    assert calls == [1, 1] and new != folder and new.name == str(first[8].date())
    assert (folder / "meta.json").exists()   # the older folder is kept


def test_saved_models_predict_the_same_after_a_reload(tmp_path):
    from features.ml_features import FEATURE_COLUMNS
    from prediction.minutes_model import LgbmMinutes
    from prediction.ml_gru import GruPoints
    from prediction.ml_points import KerasPoints, LgbmPoints
    t = _table().reset_index(drop=True)
    X, cls, order = t[FEATURE_COLUMNS], np.arange(len(t)) % 3, t["kickoff"]
    y = np.random.default_rng(0).normal(3, 2, len(t))
    seq = np.random.default_rng(1).normal(size=(len(t), 10, 11)).astype(np.float32)
    seq[..., -1] = 1
    small = dict(epochs=2, seeds=(0,))
    fitted = {"minutes": LgbmMinutes({"n_estimators": 20}).fit(X, cls, order),
              "lgbm": LgbmPoints({"n_estimators": 20}).fit(X, cls, y, order),
              "keras": KerasPoints(small).fit(X, cls, y, order),
              "gru": GruPoints(small | {"units": 4}).fit(X, seq, cls, y, order)}
    folder = tmp_path / "m"
    folder.mkdir()
    for name, model in fitted.items():
        wf._save(model, folder, name)
    for name, model in fitted.items():
        loaded = wf._load(type(model), folder, name)
        args = (X, seq) if name == "gru" else (X,)
        pred = model.predict_proba(X) if name == "minutes" else model.predict(*args)
        again = loaded.predict_proba(X) if name == "minutes" else loaded.predict(*args)
        np.testing.assert_allclose(again, pred, rtol=1e-4, atol=1e-4)


class _FakeMinutes:
    def predict_proba(self, X):
        return np.tile([0.1, 0.2, 0.7], (len(X), 1))


class _FakePoints:
    def predict(self, X, seq=None):
        return np.tile([1.0, 5.0], (len(X), 1))


def _live_inputs(flagged_chance):
    from prediction.backtest import snapshot
    snap = snapshot(synthetic_log(), "2023-24", 3)
    players = snap.players_now.assign(chance=np.where(snap.players_now["player_code"] == 100, flagged_chance, np.nan))
    return {"history": snap.history, "players_now": players, "fixtures_ahead": snap.fixtures_ahead,
            "cutoff": snap.cutoff - pd.Timedelta(days=1), "season": "2023-24",
            "teams_in_season": snap.teams_in_season}


@pytest.mark.parametrize("model", ["component", "ml-minutes", "lgbm", "keras", "gru", "stacked"])
def test_a_player_with_zero_chance_scores_zero_next_gameweek_in_every_model(model, tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "_load_models", lambda d, s: {"minutes": _FakeMinutes(), "lgbm": _FakePoints(),
                                                       "keras": _FakePoints(), "gru": _FakePoints()})
    monkeypatch.setattr(wf, "_stack_weights", lambda inputs, d: {p: np.array([0.5, 0.3, 0.2]) for p in ("GKP", "DEF", "MID", "FWD")})
    x = _live_inputs(flagged_chance=0)
    pred = wf.predict_live(model, x, tmp_path)
    first = pred["gameweek"].min()
    now = pred[pred["gameweek"] == first]
    assert now.loc[now.player_code == 100, "total"].abs().max() < 1e-9
    assert (now.loc[now.player_code != 100, "total"] > 0).all()
    later = pred[(pred["gameweek"] > first) & (pred.player_code == 100)]
    assert (later["total"] > 0).all()                                  # the flag fades with the horizon
    assert {"player_code", "gameweek", "fixture_id", "total", "p60", "no_history"} <= set(pred.columns)


def test_ml_points_total_is_the_minutes_mixture_of_the_two_point_values(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "_load_models", lambda d, s: {"minutes": _FakeMinutes(), "lgbm": _FakePoints()})
    pred = wf.predict_live("lgbm", _live_inputs(flagged_chance=np.nan), tmp_path)
    assert pred["total"].to_numpy() == pytest.approx(0.2 * 1.0 + 0.7 * 5.0)


def test_missing_kinds_are_retrained_on_the_next_run_without_waiting(tmp_path, monkeypatch):
    asked = []

    def fit(match_log, season, data_dir, now, kinds):
        asked.append(list(kinds))
        return {k: _Fake(1) for k in kinds if k != "gru" or len(asked) > 1}      # gru fails the first time
    monkeypatch.setattr(wf, "_fit_models", fit)
    _live_files(tmp_path, through=4)
    now = pd.Timestamp("2023-10-01", tz="UTC")
    folder = wf.train_live(tmp_path, now)
    assert asked == [list(wf.ALL_KINDS)] and "gru" not in json.loads((folder / "meta.json").read_text())["models"]
    assert wf.train_live(tmp_path, now) == folder and asked[1] == ["gru"]
    assert json.loads((folder / "meta.json").read_text())["models"].count("gru") == 1
    wf.train_live(tmp_path, now)
    assert len(asked) == 2                                             # complete: nothing left to retrain


def test_models_from_another_season_or_too_old_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(wf, "_fit_models", lambda *a: {k: _Fake(1) for k in wf.ALL_KINDS})
    _live_files(tmp_path, through=2)
    wf.train_live(tmp_path, pd.Timestamp("2023-10-01", tz="UTC"))
    assert wf._load_models(tmp_path, "2023-24")
    with pytest.raises(RuntimeError, match="season"):
        wf._load_models(tmp_path, "2024-25")
    _live_files(tmp_path, through=2 + wf.RETRAIN_EVERY)
    with pytest.raises(RuntimeError, match="old"):
        wf._load_models(tmp_path, "2023-24")


def test_first_gameweek_of_a_season_trains_on_the_past_seasons_only(tmp_path, monkeypatch):
    from ingestion import storage
    log = synthetic_log(n_gw=8)
    storage.save_table(log[log["season"] == "2022-23"], "match_log", tmp_path)
    storage.save_table(_table().query("season == '2022-23'"), "ml_training_table", tmp_path)
    storage.save_table(pd.DataFrame({"id": [1, 2], "finished": [False, False],
                                     "deadline_time": pd.to_datetime(["2023-08-12", "2023-08-19"], utc=True)}),
                       "gameweeks", tmp_path)
    table = wf._live_table(storage.load_table("match_log", tmp_path), "2023-24", tmp_path)
    assert set(table["season"]) == {"2022-23"} and len(table) > 0
    monkeypatch.setattr(wf, "_fit_models", lambda *a: {k: _Fake(1) for k in wf.ALL_KINDS})
    folder = wf.train_live(tmp_path, pd.Timestamp("2023-08-10", tz="UTC"))
    assert json.loads((folder / "meta.json").read_text())["through_gameweek"] == 0


def test_live_features_use_the_walk_forward_cutoff_and_features():
    from features.ml_features import FEATURE_COLUMNS, build_features
    from prediction.backtest import load_params, snapshot
    from prediction.cli import SHIPPED_PARAMS_PATH
    snap = snapshot(synthetic_log(), "2023-24", 3)
    x = {"history": snap.history, "players_now": snap.players_now, "fixtures_ahead": snap.fixtures_ahead,
         "cutoff": snap.cutoff - pd.Timedelta(days=2), "season": "2023-24", "teams_in_season": snap.teams_in_season,
         "_cache": {"models": {"minutes": _FakeMinutes()}}}
    live = wf._live_minutes(x, None)
    assert (live["cutoff"] == snap.cutoff).all()
    ref = build_features(snap, load_params(str(SHIPPED_PARAMS_PATH)).team)
    pd.testing.assert_frame_equal(live[FEATURE_COLUMNS].reset_index(drop=True), ref[FEATURE_COLUMNS])


def _stack_files(tmp_path, monkeypatch, with_live):
    from ingestion import storage
    log = synthetic_log(n_gw=8)
    old, cur = log[log["season"] == "2022-23"], log[log["season"] == "2023-24"]
    through = 3
    storage.save_table(pd.concat([old, cur[cur["gameweek"] <= through]]), "match_log", tmp_path)
    storage.save_table(pd.DataFrame({"id": range(1, 9), "finished": [g <= through for g in range(1, 9)],
                                     "deadline_time": pd.to_datetime("2023-08-12", utc=True)}), "gameweeks", tmp_path)
    actual = old.groupby(["gameweek", "player_code"], as_index=False)["points"].sum()
    base = actual.assign(season="2022-23", deadline_gw=actual["gameweek"], total=actual["points"].astype(float))[
        ["season", "deadline_gw", "player_code", "gameweek", "total"]]
    storage.save_table(base, "ml_oos_points_lgbm_2022-23", tmp_path)
    storage.save_table(base.assign(total=0.0), "ml_oos_points_gru_2022-23", tmp_path)
    storage.save_table(base.assign(total=0.0), "ml_oos_ml_minutes_totals", tmp_path)
    pos = old.drop_duplicates("player_code")[["player_code", "position"]].merge(
        base[["season", "deadline_gw", "player_code"]], on="player_code")
    pos[["season", "deadline_gw", "player_code", "position"]].to_parquet(tmp_path / "processed" / "ml_training_table.parquet")
    if with_live:   # logged live predictions: gameweeks 2 and 3 are finished, 4 is not
        act = cur.groupby(["gameweek", "player_code"], as_index=False)["points"].sum()
        rows = []
        for gw in (2, 3, 4):
            for m, scale in (("ml-minutes", 0.0), ("lgbm", 2.0), ("gru", 0.0)):
                d = act[act["gameweek"] == gw]
                rows.append(d.assign(season="2023-24", made_for_gameweek=gw, model=m, total=d["points"] * scale)[
                    ["season", "made_for_gameweek", "gameweek", "player_code", "model", "total"]])
        storage.save_table(pd.concat(rows).assign(recorded_at=pd.Timestamp("2023-09-01", tz="UTC")), "prediction_log", tmp_path)
    monkeypatch.setattr(wf, "TEST_SEASONS", ("2022-23",))
    return storage.load_table("match_log", tmp_path)


def test_live_wide_keeps_finished_gameweeks_only_and_joins_actuals(tmp_path, monkeypatch):
    match_log = _stack_files(tmp_path, monkeypatch, with_live=True)
    wide = wf._live_wide(match_log, "2023-24", tmp_path)
    assert set(wide["gameweek"]) == {2, 3} and set(wide["deadline_gw"]) == {2, 3}
    assert wide["lgbm"].to_numpy() == pytest.approx(2 * wide["actual"].to_numpy()) and wide["position"].notna().all()
    assert wide["gru"].eq(0).all()


def test_stack_weights_add_the_scored_live_rows_to_the_oos_rows(tmp_path, monkeypatch):
    match_log = _stack_files(tmp_path, monkeypatch, with_live=False)
    x = {"history": match_log, "season": "2023-24", "fixtures_ahead": pd.DataFrame({"gameweek": [4]})}
    without = wf._stack_weights(x, tmp_path)
    assert all(w[1] == pytest.approx(1.0) and w[2] == pytest.approx(0.0) for w in without.values())   # lgbm was perfect
    _stack_files(tmp_path, monkeypatch, with_live=True)
    with_live = wf._stack_weights(x, tmp_path)
    assert all(w[1] < 0.95 for w in with_live.values())          # lgbm over-predicted live: its weight falls
