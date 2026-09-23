from ingestion import cli, storage
from ingestion.fpl_client import FPLClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.requested = []

    def get(self, url, timeout):
        self.requested.append(url)
        path = url.removeprefix("https://fantasy.premierleague.com/api/")
        return FakeResponse(self.routes[path])


def test_client_builds_urls():
    session = FakeSession({"element-summary/10/": {"history": []}})
    client = FPLClient(session=session, min_interval=0)
    client.element_summary(10)
    assert session.requested == ["https://fantasy.premierleague.com/api/element-summary/10/"]


def test_cli_run_end_to_end_offline(tmp_path, bootstrap, raw_fixtures, element_summary, entry_history):
    routes = {
        "bootstrap-static/": bootstrap,
        "fixtures/": raw_fixtures,
        "element-summary/10/": element_summary,
        "element-summary/11/": {"history": []},
        "entry/42/": {
            "id": 42, "name": "Test FC", "player_first_name": "Test", "player_last_name": "Manager",
            "started_event": 1, "current_event": 2, "summary_overall_points": 110,
            "summary_overall_rank": 150, "last_deadline_bank": 5, "last_deadline_value": 1003,
            "leagues": {"classic": [{"id": 314, "name": "Overall", "entry_rank": 150, "entry_last_rank": 100}],
                        "h2h": []},
        },
        "entry/42/history/": entry_history,
        "entry/42/transfers/": [
            {"entry": 42, "event": 2, "time": "2026-08-28T10:00:00Z",
             "element_in": 10, "element_in_cost": 102, "element_out": 99, "element_out_cost": 60},
        ],
        "entry/42/event/2/picks/": {
            "active_chip": None,
            "automatic_subs": [],
            "entry_history": {"bank": 5, "value": 1003},
            "picks": [{"element": 10, "position": 1, "multiplier": 2, "is_captain": True},
                      {"element": 11, "position": 2, "multiplier": 1, "is_captain": False}],
        },
    }
    client = FPLClient(session=FakeSession(routes), min_interval=0)
    result = cli.run(history=True, entry_id=42, data_dir=tmp_path, client=client)

    assert result["current_gw"] == 2
    assert storage.load_raw(result["snapshot"], "bootstrap-static") == bootstrap
    for name in ("players", "teams", "positions", "gameweeks", "fixtures", "chips", "player_history",
                 "entry_overview", "entry_leagues", "entry_gameweeks", "entry_chips", "entry_transfers",
                 "entry_past_seasons", "entry_picks", "entry_squad"):
        assert (tmp_path / "processed" / f"{name}.parquet").exists(), name
    assert len(storage.load_table("players", tmp_path)) == 2

    state = storage.load_json("manager_state_42", tmp_path)
    assert state["free_transfers"] == 1
    assert state["bank"] == 0.5
    assert "bboost" not in state["chips_available"]
    assert {"name": "bboost", "gameweek": 1} in state["chips_used"]
    assert state["squad_selling_value"] == 10.3 + 5.0
    assert state["total_hit_points"] == 4
    assert storage.load_json("game_rules", tmp_path)["max_free_transfers"] == 5

    summary = cli.summarise(result)
    assert "Current GW: 2" in summary and "Free transfers for GW3: 1" in summary
