import numpy as np
import pandas as pd
import pytest

from evaluation.monitoring import FPL, monitoring
from ingestion import storage


def _logs(tmp_path, extra_pred=None):
    codes = [1, 2, 3, 4]
    actual = {1: 10.0, 2: 6.0, 3: 2.0, 4: 0.0}
    mins = {1: 90.0, 2: 90.0, 3: 20.0, 4: 0.0}
    match = pd.DataFrame([{"season": "2026-27", "gameweek": 6, "player_code": c, "points": actual[c],
                           "minutes": mins[c], "kickoff": pd.Timestamp("2026-10-10", tz="UTC")} for c in codes])
    storage.save_table(match, "match_log", tmp_path)
    rows = []
    for model, offset in (("ml-minutes", 0.0), ("ml-minutes+news", 1.0)):
        for c in codes:
            rows.append({"season": "2026-27", "made_for_gameweek": 6, "gameweek": 6, "player_code": c, "model": model,
                         "total": actual[c] + offset, "p60": 0.95 if mins[c] >= 60 else 0.05})
    for c in codes:   # later horizon: must not be scored
        rows.append({"season": "2026-27", "made_for_gameweek": 6, "gameweek": 7, "player_code": c,
                     "model": "ml-minutes", "total": 100.0, "p60": 0.5})
    storage.save_table(pd.DataFrame(rows), "prediction_log", tmp_path)
    ep = pd.DataFrame({"season": "2026-27", "gameweek": 6, "player_code": codes, "ep_next": [4.0, 4.0, 4.0, 4.0],
                       "recorded_at": pd.Timestamp("2026-10-09", tz="UTC")})
    storage.save_table(ep, "ep_next_log", tmp_path)


def test_rmse_and_rho_per_model(tmp_path):
    _logs(tmp_path)
    t = monitoring(tmp_path).table.set_index("model")
    assert t.loc["ml-minutes", "rmse"] == pytest.approx(0.0) and t.loc["ml-minutes", "rho"] == pytest.approx(1.0)
    assert t.loc[FPL, "n"] == 4 and t.loc[FPL, "rmse"] > 3


def test_only_next_gw_rows_scored(tmp_path):
    _logs(tmp_path)
    t = monitoring(tmp_path).table.set_index("model")
    assert t.loc["ml-minutes", "n"] == 4 and t.loc["ml-minutes", "rmse"] == pytest.approx(0.0)


def test_calibration_buckets(tmp_path):
    _logs(tmp_path)
    c = monitoring(tmp_path).calibration
    c = c[c["model"] == "ml-minutes"].set_index("bucket")
    assert c.loc[0.9, "n"] == 2 and c.loc[0.9, "observed"] == 1.0
    assert c.loc[0.0, "n"] == 2 and c.loc[0.0, "observed"] == 0.0


def test_news_model_compared_with_raw(tmp_path):
    _logs(tmp_path)
    t = monitoring(tmp_path).table.set_index("model")
    assert t.loc["ml-minutes+news", "mean_error"] == pytest.approx(1.0)
    assert t.loc["ml-minutes+news", "rmse"] > t.loc["ml-minutes", "rmse"]


def test_planner_realised_points(tmp_path):
    codes = [11, 12, 13, 14]   # codes differ from ids to exercise the remap
    storage.save_table(pd.DataFrame({"id": [1, 2, 3, 4], "code": codes, "position": ["GKP", "DEF", "MID", "FWD"]}),
                       "players", tmp_path)
    storage.save_table(pd.DataFrame({"id": [1, 2, 3, 4], "position": ["GKP", "DEF", "MID", "FWD"],
                                     "squad_select": [1, 1, 1, 1], "squad_min_play": [1, 0, 0, 0],
                                     "squad_max_play": [1, 1, 1, 1]}), "positions", tmp_path)
    storage.save_json({"starting_xi": 3, "max_per_club": 3, "hit_cost": 4, "max_free_transfers": 5}, "game_rules", tmp_path)
    pts = {11: 2.0, 12: 3.0, 13: 5.0, 14: 0.0}
    mins = {11: 90.0, 12: 90.0, 13: 90.0, 14: 0.0}
    storage.save_table(pd.DataFrame([{"season": "2026-27", "gameweek": 6, "player_code": c, "points": pts[c],
                                      "minutes": mins[c], "kickoff": pd.Timestamp("2026-10-10", tz="UTC")}
                                     for c in codes]), "match_log", tmp_path)
    storage.save_table(pd.DataFrame({"id": [6], "finished": [True]}), "gameweeks", tmp_path)
    storage.save_table(pd.DataFrame({"season": ["2026-27"], "gameweek": [6], "entry": [1], "starting": ["1,2,3"],
                                     "bench": ["4"], "captain": [3], "vice_captain": [2], "chip": [""], "hits": [1],
                                     "projected_points": [12.0]}), "plan_log", tmp_path)
    _logs(tmp_path)   # prediction_log, ep_next_log, and its own match_log are overwritten next
    storage.save_table(pd.DataFrame([{"season": "2026-27", "gameweek": 6, "player_code": c, "points": pts[c],
                                      "minutes": mins[c], "kickoff": pd.Timestamp("2026-10-10", tz="UTC")}
                                     for c in codes]), "match_log", tmp_path)
    row = monitoring(tmp_path).planner.iloc[0]
    assert row["realised"] == pytest.approx(2 + 3 + 5 + 5 - 4)   # XI + captain doubled - one hit
    assert row["projected"] == 12.0
