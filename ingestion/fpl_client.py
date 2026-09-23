"""Thin HTTP client for the official (unauthenticated) FPL API."""

from __future__ import annotations

import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fantasy.premierleague.com/api/"
USER_AGENT = "fpl-ai-coach/0.1"


class FPLClient:
    """Fetches raw JSON from the FPL API.

    Returns parsed JSON exactly as served — no transformation happens here,
    so raw snapshots stay faithful to the source.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        base_url: str = BASE_URL,
        min_interval: float = 0.2,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_request = 0.0
        self.session = session or self._default_session()

    @staticmethod
    def _default_session() -> requests.Session:
        session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        retry = Retry(
            total=5,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session

    def url(self, path: str) -> str:
        return f"{self.base_url}{path.strip('/')}/"

    def get(self, path: str) -> Any:
        # Polite rate limit: FPL has no published limit, but hammering it
        # during the ~670-request history pull gets you throttled.
        wait = self.min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        response = self.session.get(self.url(path), timeout=self.timeout)
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response.json()

    def bootstrap_static(self) -> dict:
        """Players, teams, gameweeks, positions and game settings."""
        return self.get("bootstrap-static")

    def fixtures(self) -> list[dict]:
        """All 380 fixtures for the season, with FDR per side."""
        return self.get("fixtures")

    def element_summary(self, player_id: int) -> dict:
        """One player's per-gameweek history, upcoming fixtures, past seasons."""
        return self.get(f"element-summary/{player_id}")

    def entry(self, entry_id: int) -> dict:
        """A manager's team overview (bank, value, current gameweek)."""
        return self.get(f"entry/{entry_id}")

    def entry_picks(self, entry_id: int, gameweek: int) -> dict:
        """A manager's 15 picks, captaincy and chip for a given gameweek."""
        return self.get(f"entry/{entry_id}/event/{gameweek}/picks")

    def entry_history(self, entry_id: int) -> dict:
        """Per-GW points/bank/transfers/hits this season, chips played, past seasons."""
        return self.get(f"entry/{entry_id}/history")

    def entry_transfers(self, entry_id: int) -> list[dict]:
        """Every transfer this season, with purchase (`element_in_cost`) prices."""
        return self.get(f"entry/{entry_id}/transfers")
