import numpy as np
import pandas as pd
import pytest

from prediction.stacker import MIN_GAMEWEEKS, fit_weights, stacked_oos, wide_table


def test_weights_are_non_negative_and_recover_a_blend():
    rng = np.random.default_rng(0)
    a, b = rng.normal(3, 1, 500), rng.normal(3, 1, 500)
    y = 0.7 * a + 0.3 * b
    w = fit_weights(pd.DataFrame({"a": a, "b": b, "bad": -a}), pd.Series(y))
    assert (w >= 0).all() and w[0] == pytest.approx(0.7, abs=0.02) and w[2] == pytest.approx(0, abs=1e-9)


def _frames():
    rng = np.random.default_rng(1)
    rows, acts = [], []
    for gw in range(1, 13):
        y = rng.normal(3, 1, 20)
        noise = lambda: y + rng.normal(5, 2, 20)  # noqa: E731
        for name, perfect in (("a", gw < 10), ("b", gw >= 10)):
            rows.append((name, pd.DataFrame({"season": "s", "deadline_gw": gw, "player_code": range(20), "gameweek": gw,
                                             "total": y if perfect else noise()})))
        acts.append(pd.DataFrame({"season": "s", "gameweek": gw, "player_code": range(20), "actual": y}))
    frames = {n: pd.concat([f for m, f in rows if m == n], ignore_index=True) for n in ("a", "b")}
    positions = pd.DataFrame({"season": "s", "deadline_gw": np.repeat(range(1, 13), 20), "player_code": list(range(20)) * 12,
                              "position": "MID"})
    return frames, pd.concat(acts, ignore_index=True), positions


def test_stacker_only_uses_gameweeks_played_before_the_deadline():
    frames, actuals, positions = _frames()
    out = stacked_oos(frames, actuals, positions)
    assert list(out.columns) == ["season", "deadline_gw", "player_code", "gameweek", "total"]
    at = lambda gw: out[out["deadline_gw"] == gw].set_index("player_code")["total"]  # noqa: E731
    a, b = (frames[n][frames[n]["deadline_gw"] == 10].set_index("player_code")["total"] for n in "ab")
    assert at(10).to_numpy() == pytest.approx(a.to_numpy())          # b's accuracy from gw 10 is unseen
    few = MIN_GAMEWEEKS - 1                                           # fewer than MIN_GAMEWEEKS earlier gameweeks
    a3, b3 = (frames[n][frames[n]["deadline_gw"] == few].set_index("player_code")["total"] for n in "ab")
    assert at(few).to_numpy() == pytest.approx(((a3 + b3) / 2).to_numpy())


def test_position_is_the_one_held_at_the_deadline():
    frames, actuals, positions = _frames()
    positions.loc[(positions["player_code"] == 0) & (positions["deadline_gw"] >= 6), "position"] = "DEF"
    wide = wide_table(frames, actuals, positions)
    pos = wide[wide["player_code"] == 0].set_index("deadline_gw")["position"]
    assert pos[5] == "MID" and pos[6] == "DEF" and pos[12] == "DEF"


def test_member_merge_losing_rows_raises():
    frames, actuals, positions = _frames()
    frames["b"] = frames["b"][frames["b"]["player_code"] != 3]
    frames["a"] = frames["a"][frames["a"]["player_code"] != 4]
    with pytest.raises(ValueError, match="lost rows"):
        wide_table(frames, actuals, positions)
