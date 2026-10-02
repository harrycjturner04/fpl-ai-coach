"""Poisson expectations used by the component model."""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

N_MAX = 60  # P(n >= 60) is negligible for every lambda this model produces


def expected_floor_div(lam, k: int) -> np.ndarray:
    """E[floor(N / k)] for N ~ Poisson(lam), element-wise (e.g. -1 per 2 goals conceded, 1 per 3 saves)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    n = np.arange(N_MAX)
    return ((n // k)[:, None] * poisson.pmf(n[:, None], lam[None, :])).sum(axis=0)
