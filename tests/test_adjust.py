from datetime import date

import pandas as pd
import pytest

from prediction.availability import NEWS_COLUMNS
from reasoning.adjust import adjust
from reasoning.flags import FLAG_SCHEMA, Flag, parse_flags

FADE = 0.5
NEWS_ADDED = pd.Timestamp("2026-11-03 10:00", tz="UTC")


def players(chance=None, status="a"):
    """One player, id 7, code 700."""
    return pd.DataFrame({"id": [7], "code": [700], "chance_of_playing_next_round": [chance],
                         "news_added": [NEWS_ADDED if chance is not None else pd.NaT], "status": [status]})


def pred(p60=0.8, psub=0.1, horizons=3):
    return pd.DataFrame({"player_code": 700, "horizon": range(horizons), "p60": p60, "psub": psub})


def flag(**kw):
    base = dict(player_id=7, status="out", start_prob_next_gw=0.0, return_gw=None, confidence="medium",
                source_url="https://example.com/a", source_date=date(2026, 11, 5), source_kind="media",
                summary="test")
    return Flag(**{**base, **kw})


def run(flags, pl=None, pr=None, horizon=3):
    return adjust(flags, players() if pl is None else pl, pred() if pr is None else pr, next_gw=10,
                  horizon=horizon, flag_fade=FADE)


def avail(news, h):
    return news.loc[news.horizon == h, "availability"].item()


def test_output_frame_shape():
    news, _ = run([flag()])
    assert list(news.columns) == NEWS_COLUMNS
    assert sorted(news.horizon) == [0, 1, 2]
    assert news.player_code.eq(700).all()


@pytest.mark.parametrize("chance,status", [(0, "i"), (0, "s"), (None, "s"), (None, "u"), (0, "n")])
def test_fpl_ruled_out_cannot_be_raised(chance, status):
    pl = players(chance, status)
    pl["news_added"] = NEWS_ADDED
    news, audit = run([flag(status="fit", start_prob_next_gw=1.0, confidence="high", source_kind="official")], pl)
    assert news.availability.isna().all()
    assert audit and all(a.applied == a.raw for a in audit)
    if chance == 0:  # chance NaN means raw availability 1 already, so there is nothing to raise
        assert "ruled out" in audit[0].reason


@pytest.mark.parametrize("url,changed", [("", False), ("   ", False), ("https://x.com/b", True)])
def test_lowering_needs_a_source_url(url, changed):
    news, audit = run([flag(status="doubt", start_prob_next_gw=0.75, source_url=url)])
    if changed:
        assert avail(news, 0) == pytest.approx(0.75)
    else:
        assert news.availability.isna().all() and "source_url" in audit[0].reason


@pytest.mark.parametrize("chance,src,changed", [
    (None, date(2026, 11, 5), False),       # not FPL-flagged: already at 1, nothing to raise
    (50, date(2026, 11, 2), False),         # source older than news_added
    (50, date(2026, 11, 3), False),         # same day: not strictly newer
    (50, date(2026, 11, 4), True),
])
def test_raising_needs_newer_source_than_news_added(chance, src, changed):
    news, audit = run([flag(status="fit", start_prob_next_gw=0.75 if chance else 1.0, source_date=src)],
                      players(chance, "a" if chance is None else "d"))
    if changed:
        assert avail(news, 0) == pytest.approx(0.75) and audit[0].applied == pytest.approx(0.75)
    else:
        assert news.availability.isna().all() and audit[0].applied == audit[0].raw


def test_missing_source_date_cannot_raise_but_can_lower():
    raise_, audit_r = run([flag(status="fit", start_prob_next_gw=0.75, source_date=None)], players(50, "d"))
    assert raise_.availability.isna().all() and "source_date" in audit_r[0].reason
    lower, _ = run([flag(status="doubt", start_prob_next_gw=0.25, source_date=None)], players(50, "d"))
    assert avail(lower, 0) == pytest.approx(0.25)
    # unparseable date from the LLM becomes None, same treatment
    flags, warnings = parse_flags([{**_raw_flag(), "status": "fit", "start_prob_next_gw": 0.75,
                                    "source_date": "last Tuesday"}])
    assert not warnings and flags[0].source_date is None


@pytest.mark.parametrize("conf,kind,expected", [
    ("medium", "official", 0.5), ("high", "media", 0.5), ("low", "official", 0.5), ("high", "official", 0.0)])
def test_cap_half_unless_high_official(conf, kind, expected):
    news, audit = run([flag(confidence=conf, source_kind=kind)], horizon=1)
    assert avail(news, 0) == pytest.approx(expected)
    assert audit[0].capped is (expected != 0.0)
    assert abs(audit[0].applied - audit[0].raw) <= 0.5 or (conf, kind) == ("high", "official")


def test_predicted_lineup_never_uncapped():
    news, audit = run([flag(confidence="high", source_kind="predicted_lineup")], horizon=1)
    assert avail(news, 0) == pytest.approx(0.5) and audit[0].capped
    flags, _ = parse_flags([{**_raw_flag(), "confidence": "high", "source_kind": "predicted_lineup"}])
    assert flags[0].confidence == "medium"


def test_return_gw_zeroes_weeks_before():
    news, audit = run([flag(return_gw=12, confidence="high", source_kind="official")], horizon=4)
    assert [avail(news, h) for h in range(2)] == pytest.approx([0.0, 0.0])
    assert news.loc[news.horizon >= 2, "availability"].isna().all()  # NaN: the model value is used
    assert [a.horizon for a in audit] == [0, 1]
    # capped flag: before return_gw each week is capped at raw - 0.5
    news, _ = run([flag(return_gw=12)], horizon=4)
    assert [avail(news, h) for h in range(2)] == pytest.approx([0.5, 0.5])
    assert news.loc[news.horizon >= 2, "availability"].isna().all()


def test_no_return_gw_fades():
    news, _ = run([flag(status="doubt", start_prob_next_gw=0.6)], horizon=3)
    assert [avail(news, h) for h in range(3)] == pytest.approx([0.6, 1 - 0.4 * FADE, 1 - 0.4 * FADE ** 2])
    # FPL-flagged raw values decay too; the override replaces them
    news, audit = run([flag(status="doubt", start_prob_next_gw=0.25)], players(50, "d"), horizon=2)
    assert avail(news, 1) == pytest.approx(1 - 0.75 * FADE)
    assert audit[1].raw == pytest.approx(1 - 0.5 * FADE)


def test_rotation_sets_start_share_week_one_only():
    news, audit = run([flag(status="rotation_risk", start_prob_next_gw=0.6)])
    assert news.loc[news.horizon == 0, "start_share"].item() == pytest.approx(0.6 / 0.8)
    assert news.loc[news.horizon > 0, "start_share"].isna().all()
    assert news.availability.isna().all()
    assert [(a.kind, a.horizon) for a in audit] == [("start_share", 0)]
    assert audit[0].raw == pytest.approx(0.8) and audit[0].applied == pytest.approx(0.6)
    # capped on the start probability, and raw p60 of 0 is skipped
    capped, _ = run([flag(status="rotation_risk", start_prob_next_gw=0.1)])
    assert capped.loc[capped.horizon == 0, "start_share"].item() == pytest.approx(0.3 / 0.8)
    zero, audit0 = run([flag(status="rotation_risk", start_prob_next_gw=0.1)], pr=pred(p60=0.0))
    assert zero.start_share.isna().all() and "p60" in audit0[0].reason
    # cannot raise the start share above 1
    up, _ = run([flag(status="rotation_risk", start_prob_next_gw=1.0)])
    assert up.start_share.dropna().le(1).all()


def test_every_change_is_in_audit():
    pl = pd.DataFrame({"id": [7, 8, 9], "code": [700, 800, 900], "chance_of_playing_next_round": [None, 50, 0],
                       "news_added": [pd.NaT, NEWS_ADDED, NEWS_ADDED], "status": ["a", "d", "i"]})
    pr = pd.concat([pred(), pred().assign(player_code=800), pred().assign(player_code=900)])
    flags = [flag(), flag(player_id=8, status="fit", start_prob_next_gw=0.9), flag(player_id=9, status="fit"),
             flag(player_id=123), flag(player_id=8, status="out")]
    news, audit = adjust(flags, pl, pr, next_gw=10, horizon=3, flag_fade=FADE)
    for row in news.itertuples():
        pid = {700: 7, 800: 8, 900: 9}[row.player_code]
        for kind in ("availability", "start_share"):
            v = getattr(row, kind)
            if pd.notna(v):
                assert 0 <= v <= 1
                assert any(a.player_id == pid and a.horizon == row.horizon and a.kind == kind
                           and a.applied != a.raw for a in audit)
    assert {a.player_id for a in audit} == {7, 8, 9, 123}
    assert any("duplicate" in a.reason for a in audit)
    assert any("not in players" in a.reason for a in audit)


def _raw_flag():
    return {"player_id": 7, "status": "out", "start_prob_next_gw": 0.0, "return_gw": None, "confidence": "medium",
            "source_url": "https://example.com/a", "source_date": "2026-11-05", "source_kind": "media",
            "summary": "Hamstring, out for two weeks."}


def test_parse_flags_validates():
    good = _raw_flag()
    bad = [{**good, "status": "injured"}, {**good, "start_prob_next_gw": 1.5}, {**good, "player_id": "7"},
           {**good, "extra": 1}, {k: v for k, v in good.items() if k != "summary"}, {**good, "return_gw": 2.5},
           "not a dict"]
    flags, warnings = parse_flags([good, *bad])
    assert len(flags) == 1 and flags[0].source_date == date(2026, 11, 5) and len(warnings) == len(bad)
    assert set(FLAG_SCHEMA["required"]) == set(good) and FLAG_SCHEMA["additionalProperties"] is False
    assert FLAG_SCHEMA["properties"]["return_gw"]["type"] == ["integer", "null"]
