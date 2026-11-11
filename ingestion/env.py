"""Load local secrets from the project's `.env` file (git-excluded, never committed).

Lines are `NAME=value`; blank lines and `#` comments are skipped, surrounding quotes stripped.
Variables already set in the environment win, so a terminal-level value overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        value = value.strip().strip("\"'")
        if value:  # a blank placeholder must not mask a missing key
            os.environ.setdefault(name.strip(), value)
