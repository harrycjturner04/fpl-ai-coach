import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from ingestion import congestion
from ingestion.storage import load_table

MATCHES = json.loads((Path(__file__).parent / "data" / "football_data_matches.json").read_text())["matches"]
CLUBS = [
    (3, "Arsenal", "ARS"), (7, "Aston Villa", "AVL"), (91, "Bournemouth", "BOU"), (94, "Brentford", "BRE"),
    (36, "Brighton", "BHA"), (8, "Chelsea", "CHE"), (9, "Coventry City", "COV"), (31, "Crystal Palace", "CRY"),
    (11, "Everton", "EVE"), (54, "Fulham", "FUL"), (88, "Hull City", "HUL"), (40, "Ipswich Town", "IPS"),
    (2, "Leeds", "LEE"), (14, "Liverpool", "LIV"), (43, "Man City", "MCI"), (1, "Man Utd", "MUN"),
    (4, "Newcastle", "NEW"), (17, "Nott'm Forest", "NFO"), (6, "Spurs", "TOT"), (56, "Sunderland", "SUN"),
]
TEAMS = pd.DataFrame(CLUBS, columns=["code", "name", "short_name"])


def test_maps_all_twenty_pl_clubs():
    pl = [m for m in MATCHES if m["competition"]["code"] == "PL"]
    names = {m[s]["name"] for m in pl for s in ("homeTeam", "awayTeam")}
    codes = {congestion.team_code(n, TEAMS) for n in names}
    assert len(names) == 20 and codes == set(TEAMS.code)


def test_keeps_only_non_pl_matches_of_pl_clubs():
    t = congestion.congestion_table(MATCHES, TEAMS)
    assert list(t.columns) == ["team_code", "competition", "kickoff", "opponent"]
    assert set(t.competition) == {"CL", "EL", "FAC"}
    assert len(t) == 7  # Luton v Brentford gives one row; Chelsea v Everton gives two
    row = t[(t.team_code == 3) & (t.competition == "CL")].iloc[0]
    assert row.opponent == "Real Madrid CF"


def test_near_window():
    t = congestion.congestion_table(MATCHES, TEAMS)
    ko = pd.Timestamp("2026-11-28T15:00:00Z")
    assert list(congestion.near(t, 6, ko).competition) == ["EL"]  # Spurs, 1 day before
    assert congestion.near(t, 6, ko, days=0).empty
    assert congestion.near(t, 3, pd.Timestamp("2026-11-30T15:00:00Z")).empty


class FakeSession:
    def __init__(self, status=200):
        self.status, self.calls = status, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        status = self.status

        class R:
            status_code = status
            response = None

            def json(self):
                return {"matches": MATCHES[:1]}

            def raise_for_status(self):
                if status >= 400:
                    raise requests.HTTPError(response=self)

        return R()


def test_fetch_matches_chunks_into_ten_day_windows():
    s = FakeSession()
    out = congestion.fetch_matches(date(2026, 11, 1), date(2026, 11, 25), "k", s)
    assert len(s.calls) == 3 and len(out) == 3
    assert s.calls[0][1] == {"dateFrom": "2026-11-01", "dateTo": "2026-11-10"}
    assert s.calls[2][1] == {"dateFrom": "2026-11-21", "dateTo": "2026-11-25"}
    assert s.calls[0][2] == {"X-Auth-Token": "k"}


def test_missing_key_skips_with_note(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("FOOTBALL_DATA_API_KEY", raising=False)
    assert congestion.update(tmp_path, session=FakeSession()) is None
    assert "FOOTBALL_DATA_API_KEY" in capsys.readouterr().out


@pytest.mark.parametrize("status", [429, 403])
def test_rate_limit_skips_with_note(monkeypatch, tmp_path, capsys, status):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "secret")
    assert congestion.update(tmp_path, session=FakeSession(status)) is None
    out = capsys.readouterr().out
    assert str(status) in out and "secret" not in out


def test_update_saves_table(monkeypatch, tmp_path):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "k")
    from ingestion.storage import save_table

    save_table(TEAMS, "teams", tmp_path)

    class S(FakeSession):
        def get(self, *a, **k):
            r = super().get(*a, **k)
            r.json = lambda: {"matches": MATCHES}
            return r

    t = congestion.update(tmp_path, session=S(), today=date(2026, 11, 6), days=5)
    assert not t.empty and load_table("congestion", tmp_path).equals(t)
