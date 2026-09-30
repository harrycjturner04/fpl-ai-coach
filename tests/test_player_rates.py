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


def test_dc_thresholds_from_scoring_rules():
    # Relabel the two synthetic seasons so the predicted season (2025-26) is a DC season,
    # with some of its matches before CUTOFF and some after.
    log = synthetic_log()
    log.loc[log.season == "2022-23", "season"] = "2024-25"
    log.loc[log.season == "2023-24", "season"] = "2025-26"
    def_player, mid_player, gkp_player = 101, 103, 100  # k=1 DEF, k=3 MID, k=0 GKP; all always play 90
    played60 = log.minutes >= 60
    log.loc[(log.player_code == def_player) & played60, "defensive_contribution"] = 10  # DEF threshold
    log.loc[(log.player_code == mid_player) & played60, "defensive_contribution"] = 11  # below MID threshold (12)
    log.loc[(log.player_code == gkp_player) & played60, "defensive_contribution"] = 999  # never counts for GKP

    feats, _ = player_features(log, CUTOFF, "2025-26", FLAT, PlayerParams(kappa_dc=1e-9))

    assert feats.loc[def_player].p_dc == pytest.approx(1.0, abs=1e-6)
    assert feats.loc[mid_player].p_dc == pytest.approx(0.0, abs=1e-6)
    assert feats.loc[gkp_player].p_dc == pytest.approx(0.0, abs=1e-6)
