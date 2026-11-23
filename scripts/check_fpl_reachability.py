"""Check that every FPL endpoint the pipeline uses answers from this machine.

Stage 7b runs this once as a Cloud Run Job, since FPL sometimes refuses
data-centre IPs. Standalone (only `requests`) so it runs in a bare image.

    python scripts/check_fpl_reachability.py --entry 8298351
"""
from __future__ import annotations

import argparse
import sys
import time

import requests

BASE = "https://fantasy.premierleague.com/api/"
UA = "fpl-ai-coach/0.1"  # same as ingestion/fpl_client.py


def fetch(session: requests.Session, path: str):
    started = time.monotonic()
    try:
        r = session.get(BASE + path, timeout=30)
        ok = r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json")
        detail = f"HTTP {r.status_code}, {len(r.content)} bytes"
        return ok, detail, r.json() if ok else None, time.monotonic() - started
    except requests.RequestException as exc:
        return False, type(exc).__name__, None, time.monotonic() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", type=int, required=True)
    entry = ap.parse_args().entry
    s = requests.Session()
    s.headers["User-Agent"] = UA

    failures = 0

    def check(path: str):
        nonlocal failures
        ok, detail, body, secs = fetch(s, path)
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {path:<40} {detail} ({secs:.1f}s)", flush=True)
        return body

    boot = check("bootstrap-static/") or {}
    check("fixtures/")
    players = boot.get("elements") or [{"id": 1}]
    check(f"element-summary/{players[0]['id']}/")
    check(f"entry/{entry}/")
    check(f"entry/{entry}/history/")
    check(f"entry/{entry}/transfers/")
    events = boot.get("events") or []
    gw = next((e["id"] for e in events if e.get("is_current")), None) or 1
    check(f"entry/{entry}/event/{gw}/picks/")
    check(f"event/{gw}/live/")
    print("All endpoints reachable." if not failures else f"{failures} endpoint(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
