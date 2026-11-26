"""Golden cases for the browser swap preview (`presentation/web/src/swap/swap.ts`), answered by the real Python code.

    python scripts/make_swap_golden.py

Rules come from local `data/processed` (game_rules.json, positions, plan settings' max_hits). Legality is
`optimisation.validate.check_squad`, best XI/captain/vice `optimisation.model.solve(..., max_transfers=0)` on the
post-swap squad, net transfers `orchestration.swaps.net_swaps`. `tests/test_swap_golden.py` regenerates from the
committed rules and compares, so Python and TypeScript cannot drift.

Per case `expect`: reasons in order composition ("needs a {position}"), club ("already {cap} from {team}"), budget
("£{x:.1f}m short"), hit cap; XI fields null unless legal; week_points = XI + captain bonus - hits * hit_cost;
five_week = sum over the case's weeks (hits once; Free Hit: weeks 2+ use the original squad).
"""
from __future__ import annotations

import ast
import json
import random
import re
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from ingestion import storage
from optimisation.model import InfeasibleError, Solution, SquadRules, solve
from optimisation.settings import load_settings
from optimisation.validate import check_squad
from orchestration.swaps import net_swaps

GOLDEN = Path(__file__).resolve().parent.parent / "presentation" / "web" / "src" / "swap" / "golden.json"
SEED = 7
WEEKS = 5
N_RANDOM = 28
TEAMS = ["ARS", "AVL", "BOU", "BRE", "BHA", "CHE", "CRY", "EVE", "FUL", "IPS",
         "LEI", "LIV", "MCI", "MUN", "NEW", "NFO", "SOU", "TOT", "WHU", "WOL"]

_COMPOSITION = re.compile(r"squad has (\d+) (\w+), expected (\d+)$")
_BUDGET = re.compile(r"over budget: cost ([\d.]+) > budget ([\d.]+)$")
_CLUB = "club limit exceeded: "
_BOOKKEEPING = ("hits ", "free transfers next week", "transfers in/out")


def hits_message(max_hits: int) -> str:
    return f"At most {max_hits} hits a week: play a Wildcard or Free Hit, or make fewer swaps."


def _best(players: pd.DataFrame, squad: list[int], week: int, rules: SquadRules) -> Solution:
    """Best XI/captain of a held squad for one week."""
    p = players[players["id"].isin(squad)]
    scores = pd.Series({r.id: r.points[week] for r in p.itertuples()})
    return solve(p[["id", "position", "team", "price"]], scores, rules,
                 current_squad=dict(zip(p["id"], p["price"])), bank=0.0, max_transfers=0)


def expect(case: dict, rules: SquadRules, max_hits: int) -> dict:
    players = pd.DataFrame(case["players"])
    owned = {r["id"]: r["selling_price"] for r in case["players"] if r["selling_price"] is not None}
    sold, bought = net_swaps(set(owned), [tuple(s) for s in case["swaps"]])
    squad = sorted((set(owned) - set(sold)) | set(bought))
    free, chip = case["free_transfers"], case["chip"]
    hits = 0 if chip or free is None else max(0, len(bought) - free)

    try:
        week1 = _best(players, squad, 0, rules)
    except InfeasibleError:   # composition or club cap broken: no XI, a placeholder lineup for the checker
        week1 = None
        xi = rules.starting_xi
        lineup = Solution(squad=squad, starting=squad[:xi], bench=squad[xi:], captain=squad[0], vice_captain=squad[1])
    else:
        lineup = week1
    problems = check_squad(replace(lineup, transfers_in=bought, transfers_out=sold), players, rules,
                           budget=None, current_squad=owned, bank=case["bank"])
    reasons = []
    for msg in problems:
        if m := _COMPOSITION.match(msg):
            if int(m[1]) < int(m[3]):
                reasons.append(f"needs a {m[2]}")
        elif msg.startswith(_CLUB):
            reasons += [f"already {rules.max_per_club} from {team}"
                        for team in sorted(ast.literal_eval(msg[len(_CLUB):]))]
        elif m := _BUDGET.match(msg):
            reasons.append(f"£{(round(float(m[1]) * 10) - round(float(m[2]) * 10)) / 10:.1f}m short")
        elif week1 is not None and not msg.startswith(_BOOKKEEPING):
            raise AssertionError(f"{case['name']}: unexpected problem {msg}")
    if hits > max_hits:
        reasons.append(hits_message(max_hits))

    out = {"legal": not reasons, "reasons": reasons, "hits": hits, "starting": None, "captain": None,
           "vice_captain": None, "week_points": None, "five_week": None}
    if reasons:
        return out
    original = sorted(owned)
    week_pts = [week1.projected_points] + [
        _best(players, original if chip == "freehit" else squad, w, rules).projected_points
        for w in range(1, len(case["players"][0]["points"]))]
    cost = hits * rules.hit_cost
    return out | {"starting": sorted(week1.starting), "captain": week1.captain, "vice_captain": week1.vice_captain,
                  "week_points": week_pts[0] - cost, "five_week": sum(week_pts) - cost}


# ---- case construction ----

def _player(rng, pid, position, team, owned):
    price = round(rng.uniform(4.0, 13.0), 1)
    return {"id": pid, "position": position, "team": team, "price": price,
            "selling_price": round(price - rng.choice([0.0, 0.1, 0.2]), 1) if owned else None,
            "points": [round(rng.uniform(0.5, 9.5), 3) for _ in range(WEEKS)]}


def _pool(rng, rules: SquadRules) -> list[dict]:
    """Owned ids 1-15 at distinct clubs other than ARS; three candidates per position (ids 101+) at random clubs
    other than ARS."""
    players, pid = [], 1
    for pos, n in rules.composition.items():
        for _ in range(n):
            players.append(_player(rng, pid, pos, TEAMS[pid], True))
            pid += 1
    pid = 101
    for pos in rules.composition:
        for _ in range(3):
            players.append(_player(rng, pid, pos, rng.choice(TEAMS[1:]), False))
            pid += 1
    return players


def _ids(players, position, owned):
    return [p["id"] for p in players if p["position"] == position and (p["selling_price"] is not None) == owned]


def _case(name, players, swaps, bank=1.0, free=1, chip=None):
    return {"name": name, "players": players, "bank": bank, "free_transfers": free, "chip": chip,
            "swaps": [list(s) for s in swaps]}


def _random_case(rng, rules, k):
    """Up to `max_per_club - 1` owned from ARS and some cheaper candidates from it, so random buys can hit the cap."""
    players = _pool(rng, rules)
    for p in rng.sample([p for p in players if p["selling_price"] is not None], rules.max_per_club - 1):
        p["team"] = "ARS"
    for p in players:
        if p["selling_price"] is None:
            p["price"] = round(rng.uniform(4.0, 9.0), 1)
            if rng.random() < 0.3:
                p["team"] = "ARS"
    swaps, used = [], set()
    for _ in range(rng.randint(1, 4)):
        pos = rng.choice(list(rules.composition))
        out, inn = rng.choice(_ids(players, pos, True)), rng.choice(_ids(players, pos, False))
        if not {out, inn} & used:
            swaps.append((out, inn))
            used |= {out, inn}
    free = rng.choice([None, *range(1, rules.max_free_transfers + 1)])
    chip = None if free is None else rng.choice([None, None, None, "wildcard", "freehit"])   # the app offers no chip then
    return _case(f"random {k}", players, swaps, bank=round(rng.uniform(0.0, 3.0), 1), free=free, chip=chip)


def _hand_cases(rng, rules: SquadRules, max_hits: int) -> list[dict]:
    positions = list(rules.composition)
    gk, df, md, fw = positions
    cases = []

    def base():
        return _pool(rng, rules)

    def get(players, pid):
        return next(p for p in players if p["id"] == pid)

    players = base()
    cases.append(_case("no swaps", players, []))

    # Club cap: `cap` owned from ARS, buy one more.
    for name, keep_count in (("club cap", True), ("club cap like for like", False)):
        players = base()
        ars = _ids(players, df, True)[:rules.max_per_club]
        for pid in ars:
            get(players, pid)["team"] = "ARS"
        inn = _ids(players, df, False)[0]
        get(players, inn).update(team="ARS", price=4.0)
        out = _ids(players, df, True)[rules.max_per_club] if keep_count else ars[0]
        cases.append(_case(name, players, [(out, inn)], bank=10.0))

    # Budget edges: bank + selling price of the sold player vs price of the bought one.
    for name, extra in (("exact budget", 0.0), ("over budget by 0.4", 0.4), ("under budget by 0.1", -0.1)):
        players = base()
        out, inn = _ids(players, md, True)[0], _ids(players, md, False)[0]
        get(players, out)["selling_price"] = 6.0
        get(players, inn)["price"] = round(6.5 + extra, 1)
        cases.append(_case(name, players, [(out, inn)], bank=0.5))

    players = base()
    out, inn = _ids(players, df, True)[0], _ids(players, md, False)[0]
    get(players, inn)["price"] = 4.0
    cases.append(_case("wrong position", players, [(out, inn)], bank=10.0))

    players = base()          # over budget and over the club cap at once
    for pid in _ids(players, md, True)[:rules.max_per_club]:
        get(players, pid)["team"] = "ARS"
    out, inn = _ids(players, fw, True)[0], _ids(players, fw, False)[0]
    get(players, out)["selling_price"] = 5.0
    get(players, inn).update(team="ARS", price=12.3)
    cases.append(_case("club cap and over budget", players, [(out, inn)], bank=0.0))

    # Formations: weak positions get as few starters as the limits allow.
    for name, weak in (("formation fewest defenders", [df]), ("formation fewest midfielders", [md]),
                       ("formation fewest forwards", [fw]), ("formation fewest defenders and forwards", [df, fw])):
        players = base()
        for p in players:
            if p["position"] in weak:
                p["points"] = [round(rng.uniform(0.1, 0.4), 3) for _ in range(WEEKS)]
        cases.append(_case(name, players, []))

    players = base()
    keepers = _ids(players, gk, True)
    get(players, keepers[0])["points"] = [0.6, 0.7, 0.8, 0.9, 1.1]
    get(players, keepers[1])["points"] = [9.9, 9.8, 9.7, 9.6, 9.5]
    cases.append(_case("goalkeeper choice", players, []))

    players = base()
    out = _ids(players, md, True)[0]
    get(players, out)["points"] = [9.95, 0.2, 0.3, 0.4, 0.5]
    inn = _ids(players, md, False)[0]
    get(players, inn)["points"] = [0.25, 9.6, 9.7, 9.8, 9.9]
    cases.append(_case("week one star sold", players, [(out, inn)], bank=10.0))

    # Blank week: one club scores nothing in week two (owned DEF and MID; a FWD candidate from it).
    for name, buy in (("blank week", False), ("blank week buy", True)):
        players = base()
        d, m = _ids(players, df, True)[0], _ids(players, md, True)[0]
        f_out, f_in = _ids(players, fw, True)[0], _ids(players, fw, False)[0]
        for pid in (d, m, f_in):
            get(players, pid)["team"] = "BLA"
        get(players, f_in)["price"] = 4.0
        for p in players:
            if p["team"] == "BLA":
                p["points"][1] = 0.0
        cases.append(_case(name, players, [(f_out, f_in)] if buy else [], bank=10.0))

    # Hits and chips: the same swaps under different free transfers and chips.
    players = base()
    pairs = [(o, i) for pos in positions[1:] for o, i in zip(_ids(players, pos, True), _ids(players, pos, False))]
    for p in players:
        if p["selling_price"] is None:
            p["price"] = 4.0
    for name, n, free, chip in (
        ("one free transfer", 1, 1, None), ("one hit", 2, 1, None), ("banked transfers", 3, 3, None),
        ("max hits allowed", 1 + max_hits, 1, None), ("too many hits", 2 + max_hits, 1, None),
        ("free none", 2 + max_hits, None, None), ("wildcard", 2 + max_hits, 1, "wildcard"),
        ("free hit", 2 + max_hits, 1, "freehit"),
    ):
        cases.append(_case(name, players, pairs[:n], bank=10.0, free=free, chip=chip))

    a, b = pairs[0]
    c = _ids(players, get(players, a)["position"], False)[1]
    cases.append(_case("chained swap", players, [(a, b), (b, c)], bank=10.0))
    cases.append(_case("cancelled swap", players, [(a, b), (b, a)], bank=10.0))
    cases.append(_case("cancelled swap with hit", players, [(a, b), (b, a), *pairs[1:3]], bank=10.0))
    return cases


def build(rules: SquadRules, max_hits: int) -> dict:
    rng = random.Random(SEED)
    cases = _hand_cases(rng, rules, max_hits) + [_random_case(rng, rules, k) for k in range(1, N_RANDOM + 1)]
    for c in cases:
        c["expect"] = expect(c, rules, max_hits)
    return {"rules": asdict(rules) | {"max_hits": max_hits}, "cases": cases}


def render(golden: dict) -> str:
    """One case per line: compact, and a diff shows which case changed."""
    cases = ",\n".join(json.dumps(c, ensure_ascii=False) for c in golden["cases"])
    return f'{{"rules": {json.dumps(golden["rules"])},\n"cases": [\n{cases}\n]}}\n'


def main() -> None:
    rules = SquadRules.from_data(storage.load_json("game_rules"), storage.load_table("positions"))
    golden = build(rules, load_settings(storage.DATA_DIR).max_hits)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(render(golden), encoding="utf-8", newline="\n")
    legal = sum(c["expect"]["legal"] for c in golden["cases"])
    print(f"{len(golden['cases'])} cases ({legal} legal) -> {GOLDEN}")


if __name__ == "__main__":
    main()
