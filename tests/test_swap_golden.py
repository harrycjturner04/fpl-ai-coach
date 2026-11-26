import json
from dataclasses import asdict

import pytest

from ingestion import storage
from optimisation.model import SquadRules
from optimisation.settings import load_settings
from scripts.make_swap_golden import GOLDEN, build, render


def _committed():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _rules(block):
    return SquadRules(**{k: v for k, v in block.items() if k != "max_hits"}), block["max_hits"]


def test_golden_is_fresh():
    rules, max_hits = _rules(_committed()["rules"])
    assert render(build(rules, max_hits)) == GOLDEN.read_text(encoding="utf-8")


def test_golden_rules_match_data():
    if storage.load_table_or_none("positions") is None:
        pytest.skip("no local data/processed")
    rules = SquadRules.from_data(storage.load_json("game_rules"), storage.load_table("positions"))
    assert _committed()["rules"] == asdict(rules) | {"max_hits": load_settings(storage.DATA_DIR).max_hits}


def test_golden_covers_the_edge_cases():
    g = _committed()
    cap, max_hits = g["rules"]["max_per_club"], g["rules"]["max_hits"]
    cases = {c["name"]: c for c in g["cases"]}
    assert len(cases) == len(g["cases"]) >= 40
    ex = {n: c["expect"] for n, c in cases.items()}
    assert ex["exact budget"]["legal"] and ex["over budget by 0.4"]["reasons"] == ["£0.4m short"]
    assert ex["club cap"]["reasons"] == [f"already {cap} from ARS"]
    assert ex["wrong position"]["reasons"] == ["needs a DEF"]
    assert ex["too many hits"]["reasons"] == [
        f"At most {max_hits} hits a week: play a Wildcard or Free Hit, or make fewer swaps."]
    assert ex["wildcard"]["hits"] == 0 and ex["free none"]["hits"] == 0 and ex["one hit"]["hits"] == 1
    assert cases["free hit"]["swaps"] == cases["wildcard"]["swaps"] and cases["free hit"]["chip"] == "freehit"
    assert ex["free hit"]["week_points"] == ex["wildcard"]["week_points"]
    assert ex["free hit"]["five_week"] != ex["wildcard"]["five_week"]
    for c in g["cases"]:
        e = c["expect"]
        assert e["legal"] == (not e["reasons"]) and (e["starting"] is None) == (not e["legal"])
