import numpy as np, pandas as pd
from features.ml_features import FEATURE_COLUMNS, build_features
from features.team_ratings import TeamParams
from prediction.backtest import snapshot
from tests.test_backtest import synthetic_log

def test_features_ignore_rows_after_the_cutoff():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    a = build_features(snap, TeamParams())
    future = log["kickoff"] >= snap.cutoff
    log2 = log.copy(); log2.loc[future, ["minutes", "points", "xg", "bps"]] = 99
    b = build_features(snapshot(log2, "2023-24", 4), TeamParams())
    pd.testing.assert_frame_equal(a[FEATURE_COLUMNS], b[FEATURE_COLUMNS])

def test_new_player_gets_rows_with_nan_history():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    new = snap.players_now.iloc[[0]].assign(player_code=999)
    snap.players_now = pd.concat([snap.players_now, new], ignore_index=True)
    f = build_features(snap, TeamParams())
    row = f[f["player_code"] == 999]
    assert len(row) > 0 and row["new_to_league"].eq(1).all() and row["minutes_pm5"].isna().all()

def test_double_gameweek_gives_one_row_per_fixture():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    extra = snap.fixtures_ahead.iloc[[0]].assign(fixture_id=9999)       # same gameweek, second fixture
    snap.fixtures_ahead = pd.concat([snap.fixtures_ahead, extra], ignore_index=True)
    f = build_features(snap, TeamParams())
    team = extra["home_code"].iloc[0]; gw = extra["gameweek"].iloc[0]
    rows = f[(f["team_code"] == team) & (f["gameweek"] == gw)]
    assert rows.groupby("player_code")["fixture_id"].nunique().eq(2).all()
    assert rows["n_fixtures_gw"].eq(2).all()

def test_blank_club_keeps_later_fixtures():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    gw0 = snap.fixtures_ahead["gameweek"].min()
    blank = snap.fixtures_ahead.iloc[0][["home_code"]].iloc[0]
    keep = ~((snap.fixtures_ahead["gameweek"] == gw0) &
             ((snap.fixtures_ahead["home_code"] == blank) | (snap.fixtures_ahead["away_code"] == blank)))
    snap.fixtures_ahead = snap.fixtures_ahead[keep]
    f = build_features(snap, TeamParams())
    club = f[f["team_code"] == blank]
    assert (club["gameweek"] > gw0).all() and len(club) > 0

def test_moved_player_uses_new_club_for_team_features():
    log = synthetic_log()
    snap = snapshot(log, "2023-24", 4)
    pc = snap.players_now["player_code"].iloc[0]
    other = next(t for t in snap.teams_in_season if t != snap.players_now["team_code"].iloc[0])
    snap.players_now.loc[snap.players_now["player_code"] == pc, "team_code"] = other
    f = build_features(snap, TeamParams())
    assert f.loc[f["player_code"] == pc, "team_code"].eq(other).all()
    assert f.loc[f["player_code"] == pc, "new_to_club"].eq(1).all()

def test_first_gameweek_of_a_season_builds():
    log = synthetic_log()
    f = build_features(snapshot(log, "2023-24", 1), TeamParams())
    assert len(f) > 0 and f["matches_this_season"].eq(0).all()
