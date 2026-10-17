import numpy as np, pandas as pd
from features.team_ratings import TeamParams
from prediction.ml_data import TARGET_STATS, deadline_rows, labelled, minutes_class, target_points
from tests.test_backtest import synthetic_log

def test_minutes_class():
    assert list(minutes_class(np.array([0, 1, 59, 60, 90]))) == [0, 1, 1, 2, 2]

def test_target_points_match_archive_points_under_each_seasons_rules():
    row = {f"t_{c}": 0 for c in TARGET_STATS} | {"t_minutes": 90, "t_goals": 1, "t_position": "GKP"}
    t = pd.DataFrame([row])
    assert target_points(t, "2024-25").iloc[0] == 2 + 6
    assert target_points(t, "2025-26").iloc[0] == 2 + 10

def test_deadline_rows_join_targets_and_respect_cutoff():
    rows = deadline_rows(synthetic_log(), "2023-24", 3, TeamParams())
    assert (rows["kickoff"] >= rows["cutoff"]).all()
    assert rows["t_minutes"].notna().any()

def test_labelled_only_returns_played_fixtures():
    rows = deadline_rows(synthetic_log(), "2023-24", 3, TeamParams())
    cut = rows["kickoff"].sort_values().iloc[len(rows) // 2]
    got = labelled(rows, cut)
    assert len(got) and (got["kickoff"] < cut).all() and got["t_minutes"].notna().all()

def test_training_table_keeps_prior_season_history(tmp_path):
    from prediction.ml_data import training_table
    t = training_table(synthetic_log(), ["2022-23", "2023-24"], workers=1, data_dir=tmp_path)
    row = t[(t["season"] == "2023-24") & (t["deadline_gw"] == 1) & (t["player_code"] == 100)]
    assert len(row) and (row["matches_total"] > 0).all()


def test_target_points_on_a_table_with_feature_and_target_position_scores_dc_by_season():
    row = {f"t_{c}": 0 for c in TARGET_STATS} | {"t_minutes": 90, "t_defensive_contribution": 12,
                                                  "t_position": "DEF", "position": "DEF"}
    t = pd.DataFrame([row])
    assert target_points(t, "2024-25").iloc[0] == 2
    assert target_points(t, "2025-26").iloc[0] == 2 + 2
