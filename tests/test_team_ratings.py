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
