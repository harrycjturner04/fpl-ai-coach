import numpy as np
import pandas as pd
import pytest

from prediction.stacker import MIN_GAMEWEEKS, fit_weights, stacked_oos


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
    return frames, pd.concat(acts, ignore_index=True), pd.Series("MID", index=range(20))


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
