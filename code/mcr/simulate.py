"""Monte Carlo simulation of absorbing chains for validation.

Samples independent trajectories until absorption or horizon exhaustion,
returns empirical P[absorbed in success] with a Wilson CI.
"""
from __future__ import annotations

import numpy as np


def _absorbing_block(Q: np.ndarray, R_succ: np.ndarray, R_fail: np.ndarray) -> tuple:
    """Assemble the full (m+2) transition matrix and check row sums."""
    m = Q.shape[0]
    R = np.column_stack([R_succ.reshape(m), R_fail.reshape(m)])
    row_sum = Q.sum(axis=1) + R.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-8):
        raise ValueError(f"Rows of [Q | R] must sum to 1 (max |dev|={abs(row_sum-1).max():.2e})")
    P_top = np.hstack([Q, R])                           # (m, m+2)
    P_bot = np.hstack([np.zeros((2, m)), np.eye(2)])    # (2, m+2)
    return np.vstack([P_top, P_bot]), m


def monte_carlo_reliability(
    Q: np.ndarray,
    R_succ: np.ndarray,
    R_fail: np.ndarray,
    s0: int = 0,
    d: int | None = None,
    n: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Empirical estimate of R(d) by sampling n trajectories.

    If d is None, horizon is effectively infinite (continue until absorption).

    Returns {'mean': p_hat, 'lo': wilson_lo, 'hi': wilson_hi, 'n': n}.
    """
    if rng is None:
        rng = np.random.default_rng()
    Q = np.asarray(Q, dtype=float)
    R_succ = np.asarray(R_succ, dtype=float).reshape(-1)
    R_fail = np.asarray(R_fail, dtype=float).reshape(-1)
    P, m = _absorbing_block(Q, R_succ, R_fail)
    SUCC_IDX = m
    FAIL_IDX = m + 1

    # Precompute cumulative distributions per state for inverse-CDF sampling
    cumP = np.cumsum(P, axis=1)

    successes = 0
    horizon = d if d is not None else 10_000  # safety cap for "infinity"
    for _ in range(n):
        state = s0
        for _ in range(horizon):
            if state == SUCC_IDX:
                successes += 1
                break
            if state == FAIL_IDX:
                break
            u = rng.random()
            # argmax of cumP[state] > u
            state = int(np.searchsorted(cumP[state], u))
        else:
            # Exited loop without break — horizon exhausted without absorption.
            pass

    p_hat = successes / n
    z = 1.959963984540054  # 95% normal
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return {
        "mean": p_hat,
        "lo": float(centre - half),
        "hi": float(centre + half),
        "n": n,
    }
