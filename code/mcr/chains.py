"""Random substochastic chain generators for simulation studies."""
from __future__ import annotations

import numpy as np


def random_substochastic(
    m: int,
    rho_target: float = 0.7,
    density: float = 1.0,
    succ_rate: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random (Q, R_succ, R_fail) triple with rho(Q) approx rho_target.

    Strategy:
      1. Sample a random non-negative matrix M with the desired density.
      2. Row-normalize.
      3. Scale by rho_target (a bit sloppy w.r.t. exact spectral radius, but
         within ~10% for reasonably sized chains).
      4. Distribute remaining (1 - rho_target) row mass between succ and fail
         proportionally to succ_rate.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Base matrix
    if density < 1.0:
        mask = (rng.random((m, m)) < density).astype(float)
    else:
        mask = np.ones((m, m))
    raw = rng.random((m, m)) * mask
    # Row-normalize (if any row is all-zero, assign mass to self-loop to avoid NaN)
    row_sums = raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    Qhat = raw / row_sums
    # Shrink so that row sums == rho_target
    Q = Qhat * rho_target
    # Remaining mass per row
    remaining = 1.0 - Q.sum(axis=1)   # equals 1 - rho_target (approx)
    R_succ = remaining * succ_rate
    R_fail = remaining - R_succ
    return Q, R_succ, R_fail


def nhpp_scaling_family(
    m: int,
    eps_seq: np.ndarray,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return a family of chains scaling toward the NHPP rare-failure regime.

    Each chain has failure-hazard eps per step, success-hazard 0 (absorption
    only via horizon or re-tries). Corresponds to a birth-death chain with
    near-constant per-step failure rate, which is the canonical NHPP limit.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = []
    for eps in eps_seq:
        # Forward-moving chain: state i -> i+1 with prob 1 - eps, -> fail with prob eps.
        # Final state m-1 -> succ with prob 1 - eps, -> fail with prob eps.
        Q = np.zeros((m, m))
        R_succ = np.zeros(m)
        R_fail = np.zeros(m)
        for i in range(m - 1):
            Q[i, i + 1] = 1.0 - eps
            R_fail[i] = eps
        R_succ[m - 1] = 1.0 - eps
        R_fail[m - 1] = eps
        out.append((Q, R_succ, R_fail))
    return out


def deterministic_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The 3x3 example in the notation sheet (used for type-checking)."""
    Q = np.array([
        [0.5, 0.3, 0.0],
        [0.0, 0.4, 0.3],
        [0.0, 0.0, 0.6],
    ])
    R_succ = np.array([0.1, 0.2, 0.3])
    R_fail = np.array([0.1, 0.1, 0.1])
    return Q, R_succ, R_fail
