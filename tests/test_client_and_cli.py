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


def test_cli_run_end_to_end_offline(tmp_path, bootstrap, raw_fixtures, element_summary):
    routes = {
        "bootstrap-static/": bootstrap,
        "fixtures/": raw_fixtures,
        "element-summary/10/": element_summary,
        "element-summary/11/": {"history": []},
        "entry/42/": {"id": 42},
        "entry/42/event/2/picks/": {
            "active_chip": None,
            "entry_history": {"bank": 5, "value": 1000},
            "picks": [{"element": 10, "position": 1, "multiplier": 2}],
        },
    }
    client = FPLClient(session=FakeSession(routes), min_interval=0)
    result = cli.run(history=True, entry_id=42, data_dir=tmp_path, client=client)

    assert result["current_gw"] == 2
    assert storage.load_raw(result["snapshot"], "bootstrap-static") == bootstrap
    for name in ("players", "teams", "positions", "gameweeks", "fixtures", "player_history", "entry_picks"):
        assert (tmp_path / "processed" / f"{name}.parquet").exists(), name
    assert len(storage.load_table("players", tmp_path)) == 2
    assert "Current GW: 2" in cli.summarise(result)
