"""Tool-error perturbation (Definition 4).

Given (Q0, R_succ_0, R_fail_0) and a perturbation direction Delta,
produce (Q(eps), R_succ, R_fail(eps)) that re-routes eps-mass from
successful transitions to the failure absorbing state.
"""
from __future__ import annotations

import numpy as np


def perturb(
    Q0: np.ndarray,
    R_succ_0: np.ndarray,
    R_fail_0: np.ndarray,
    eps: float,
    delta: np.ndarray,
    reroute_to: str = "fail",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Q(eps) = Q0 + eps*delta, reroute outflow to R_fail (or R_succ).

    Requirements on delta:
        delta.shape == Q0.shape
        (delta @ 1).sum over columns <= 0   (removes mass from Q rows)
    The removed row-mass is then added to R_fail (reroute_to='fail') or R_succ.
    """
    Q0 = np.asarray(Q0, dtype=float)
    R_succ_0 = np.asarray(R_succ_0, dtype=float).reshape(-1)
    R_fail_0 = np.asarray(R_fail_0, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float)
    if delta.shape != Q0.shape:
        raise ValueError("delta must have same shape as Q0")
    Q_new = Q0 + eps * delta
    # Row-mass removed from Q:
    dm = -(eps * delta.sum(axis=1))   # positive if delta removes mass
    if reroute_to == "fail":
        R_fail_new = R_fail_0 + dm
        R_succ_new = R_succ_0.copy()
    elif reroute_to == "succ":
        R_fail_new = R_fail_0.copy()
        R_succ_new = R_succ_0 + dm
    else:
        raise ValueError("reroute_to must be 'fail' or 'succ'")
    # Validate substochasticity
    row_sum = Q_new.sum(axis=1) + R_succ_new + R_fail_new
    if not np.allclose(row_sum, 1.0, atol=1e-8):
        raise ValueError(
            f"Row sums deviate: max |dev|={abs(row_sum-1).max():.2e}; "
            f"check delta definition or reduce eps."
        )
    if (Q_new < -1e-12).any() or (R_succ_new < -1e-12).any() or (R_fail_new < -1e-12).any():
        raise ValueError("Negative entries after perturbation; reduce eps.")
    return Q_new, R_succ_new, R_fail_new
