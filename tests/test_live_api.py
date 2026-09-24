"""Live checks that the FPL API still serves every field this app relies on.

Skipped by default (they hit the real API). Run at the start of each season,
or whenever ingestion breaks:  pytest -m live
"""

import pytest

from ingestion.fpl_client import FPLClient

pytestmark = pytest.mark.live
PUBLIC_ENTRY = 1  # a long-standing public team


@pytest.fixture(scope="module")
def client():
    return FPLClient()


@pytest.fixture(scope="module")
def bootstrap(client):
    return client.bootstrap_static()


def _missing(record: dict, fields: set[str]) -> set[str]:
    return fields - set(record)


def test_bootstrap_sections(bootstrap):
    assert _missing(bootstrap, {"elements", "teams", "events", "element_types", "chips", "game_settings"}) == set()
    assert _missing(bootstrap["game_settings"], {
        "squad_squadsize", "squad_squadplay", "squad_team_limit", "squad_total_spend",
        "transfers_sell_on_fee", "max_extra_free_transfers"}) == set()
    assert _missing(bootstrap["element_types"][0], {
        "id", "singular_name_short", "squad_select", "squad_min_play", "squad_max_play"}) == set()
    assert _missing(bootstrap["events"][0], {"id", "deadline_time", "is_current", "is_next"}) == set()
    assert _missing(bootstrap["teams"][0], {"id", "name", "short_name"}) == set()
    assert _missing(bootstrap["chips"][0], {"id", "name", "start_event", "stop_event"}) == set()


def test_player_fields_used_by_scoring_and_optimiser(bootstrap):
    assert _missing(bootstrap["elements"][0], {
        "id", "web_name", "team", "element_type", "now_cost", "cost_change_start", "status",
        "can_select", "chance_of_playing_next_round", "ep_next", "form", "news_added"}) == set()


def test_fixture_fields(client):
    assert _missing(client.fixtures()[0], {
        "id", "event", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty",
        "team_h_score", "team_a_score"}) == set()


def test_player_history_fields(client, bootstrap):
    summary = client.element_summary(bootstrap["elements"][0]["id"])
    assert _missing(summary, {"history"}) == set()
    if summary["history"]:
        assert _missing(summary["history"][0], {"element", "round", "value", "kickoff_time"}) == set()


def test_entry_endpoints(client):
    assert _missing(client.entry(PUBLIC_ENTRY), {
        "id", "name", "player_first_name", "player_last_name", "started_event",
        "summary_overall_points", "summary_overall_rank", "leagues"}) == set()
    history = client.entry_history(PUBLIC_ENTRY)
    assert _missing(history, {"current", "chips", "past"}) == set()
    assert _missing(history["current"][0], {
        "event", "bank", "value", "event_transfers", "event_transfers_cost"}) == set()
    transfers = client.entry_transfers(PUBLIC_ENTRY)
    if transfers:
        assert _missing(transfers[0], {"event", "time", "element_in", "element_in_cost"}) == set()
    picks = client.entry_picks(PUBLIC_ENTRY, history["current"][-1]["event"])
    assert _missing(picks, {"picks", "entry_history", "active_chip"}) == set()
    assert _missing(picks["picks"][0], {"element", "position", "multiplier", "is_captain"}) == set()
