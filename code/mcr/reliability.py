"""Closed-form reliability for absorbing Markov chains.

Implements Theorem 1 (T1):

    R(d)   = e_{s0}^T (I - Q^d) N R_succ,      N = (I - Q)^{-1}
    R(inf) = e_{s0}^T          N R_succ
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import inv, matrix_power


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    """Return N = (I - Q)^{-1}. Raises if (I - Q) is singular."""
    Q = np.asarray(Q, dtype=float)
    m = Q.shape[0]
    if Q.shape != (m, m):
        raise ValueError(f"Q must be square; got shape {Q.shape}")
    I = np.eye(m)
    try:
        return inv(I - Q)
    except np.linalg.LinAlgError as e:
        raise ValueError("(I - Q) is singular; Q may not be transient") from e


def asymptotic_reliability(
    Q: np.ndarray, R_succ: np.ndarray, s0: int | np.ndarray = 0
) -> float:
    """Return R_infinity = e_{s0}^T N R_succ  (T1, eq. 2)."""
    Q = np.asarray(Q, dtype=float)
    R_succ = np.asarray(R_succ, dtype=float).reshape(-1)
    if R_succ.shape[0] != Q.shape[0]:
        raise ValueError("R_succ length must equal |S_T|")
    N = fundamental_matrix(Q)
    NR = N @ R_succ
    if np.isscalar(s0) or np.ndim(s0) == 0:
        return float(NR[int(s0)])
    pi0 = np.asarray(s0, dtype=float).reshape(-1)
    if pi0.shape[0] != Q.shape[0]:
        raise ValueError("pi0 length must equal |S_T|")
    return float(pi0 @ NR)


def reliability(
    Q: np.ndarray,
    R_succ: np.ndarray,
    s0: int | np.ndarray = 0,
    d: int | None = None,
) -> float:
    """Return R(d) when d is given, else R_infinity.

    Closed form (T1):
        R(d) = e_{s0}^T (I - Q^d) N R_succ
    """
    if d is None:
        return asymptotic_reliability(Q, R_succ, s0)
    Q = np.asarray(Q, dtype=float)
    R_succ = np.asarray(R_succ, dtype=float).reshape(-1)
    if d < 0:
        raise ValueError("d must be non-negative")
    if d == 0:
        return 0.0
    m = Q.shape[0]
    I = np.eye(m)
    N = fundamental_matrix(Q)
    Qd = matrix_power(Q, d)
    coeff = (I - Qd) @ N
    NR = coeff @ R_succ
    if np.isscalar(s0) or np.ndim(s0) == 0:
        return float(NR[int(s0)])
    pi0 = np.asarray(s0, dtype=float).reshape(-1)
    return float(pi0 @ NR)


def reliability_curve(
    Q: np.ndarray,
    R_succ: np.ndarray,
    s0: int | np.ndarray = 0,
    d_max: int = 50,
) -> np.ndarray:
    """Return R(d) for d = 0, 1, ..., d_max as a 1-D array.

    Uses repeated multiplication rather than power for efficiency:
    R(d+1) = R(d) + e_{s0}^T Q^d R_succ.
    """
    Q = np.asarray(Q, dtype=float)
    R_succ = np.asarray(R_succ, dtype=float).reshape(-1)
    m = Q.shape[0]
    if np.isscalar(s0) or np.ndim(s0) == 0:
        e = np.zeros(m)
        e[int(s0)] = 1.0
    else:
        e = np.asarray(s0, dtype=float).reshape(-1)
    out = np.zeros(d_max + 1)
    # e_t = e^T Q^t, recomputed each step
    e_t = e.copy()
    for t in range(d_max):
        out[t + 1] = out[t] + float(e_t @ R_succ)
        e_t = e_t @ Q
    return out
