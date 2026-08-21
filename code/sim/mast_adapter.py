"""mast_adapter.py — MAST CSV → (Q, R_succ, R_fail, s0) converter.

The MAST dataset (Cemri et al., 2025; public release) contains annotated
agent traces with step-level category labels across 7 frameworks.
This module:
  1. Loads a MAST-format CSV (or the canonical traces JSON).
  2. Maps step labels to a compact state space S_T.
  3. Counts transition frequencies and computes empirical Q̂ and R̂.
  4. Returns (Q, R_succ, R_fail, s0, state_names) ready for mcr.reliability.

Expected CSV columns (MAST trace format):
    trace_id   : unique identifier per trajectory
    step_idx   : 0-indexed step number within the trace
    state      : step-level category label (string)
    outcome    : 'success' | 'failure' | 'continue'

If your MAST download has a different schema, pass column_map=... to adapt.

Usage:
    from mcr.mast_adapter import load_mast_csv, build_chain
    df = load_mast_csv("data/mast_derived/mast_traces.csv")
    Q, R_succ, R_fail, s0, names = build_chain(df)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

_DEFAULT_COLS = {
    "trace_id": "trace_id",
    "step_idx": "step_idx",
    "state": "state",
    "outcome": "outcome",
}

SUCCESS_OUTCOMES = {"success", "solved", "correct", "pass"}
FAILURE_OUTCOMES = {"failure", "failed", "error", "timeout", "incorrect"}


def load_mast_csv(path: str | Path, column_map: dict[str, str] | None = None) -> Any:
    """Load a MAST trace CSV into a pandas DataFrame.

    Args:
        path: Path to the CSV file.
        column_map: Optional rename map {expected_col: actual_col}.

    Returns:
        pandas DataFrame with columns: trace_id, step_idx, state, outcome.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas is required for load_mast_csv. Install with: pip install pandas")
    df = pd.read_csv(path)
    if column_map:
        df = df.rename(columns={v: k for k, v in column_map.items()})
    required = set(_DEFAULT_COLS.keys())
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}. "
            f"Use column_map to map your column names to the expected names."
        )
    return df[list(required)]


def build_chain(
    df: Any,
    framework_filter: str | None = None,
    smoothing: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Build an empirical absorbing Markov chain from MAST trace data.

    Args:
        df: DataFrame from load_mast_csv (or compatible).
        framework_filter: If given, filter to traces from this framework
                          (assumes a 'framework' column exists).
        smoothing: Laplace smoothing count added to each transition cell.

    Returns:
        (Q̂, R̂_succ, R̂_fail, s0_index, state_names)
        Q̂[i,j] = empirical probability of moving from state i to state j.
        s0_index = index of the most common initial state.
    """
    if not HAS_PANDAS:
        raise ImportError("pandas required")

    if framework_filter is not None and "framework" in df.columns:
        df = df[df["framework"] == framework_filter].copy()

    if df.empty:
        raise ValueError("No traces after filtering.")

    # Identify unique transient states (exclude success/failure absorbing states)
    all_states = sorted(df["state"].unique().tolist())
    state_to_idx = {s: i for i, s in enumerate(all_states)}
    m = len(all_states)

    # Count matrices
    trans_counts = np.full((m, m), smoothing)
    succ_counts = np.full(m, smoothing)
    fail_counts = np.full(m, smoothing)

    for tid, trace in df.groupby("trace_id"):
        trace = trace.sort_values("step_idx")
        states_list = trace["state"].tolist()
        outcomes_list = trace["outcome"].tolist()

        for step_i, (src_label, outcome) in enumerate(zip(states_list, outcomes_list)):
            src = state_to_idx[src_label]
            if outcome.lower() in SUCCESS_OUTCOMES:
                succ_counts[src] += 1
            elif outcome.lower() in FAILURE_OUTCOMES:
                fail_counts[src] += 1
            elif step_i + 1 < len(states_list):
                # 'continue' — record transition to next state
                dst_label = states_list[step_i + 1]
                dst = state_to_idx[dst_label]
                trans_counts[src, dst] += 1

    # Row normalize
    row_totals = trans_counts.sum(axis=1) + succ_counts + fail_counts
    Q = trans_counts / row_totals[:, np.newaxis]
    R_succ = succ_counts / row_totals
    R_fail = fail_counts / row_totals

    # s0: most common first state across traces
    first_states = df.groupby("trace_id").apply(
        lambda g: g.sort_values("step_idx").iloc[0]["state"]
    )
    most_common = first_states.value_counts().idxmax()
    s0 = state_to_idx[most_common]

    return Q, R_succ, R_fail, s0, all_states


def build_synthetic_mast_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, int, list[str]]:
    """Return a synthetic chain mimicking a typical MAST framework.

    Used for testing and as a worked example in the paper when the actual
    MAST dataset is not available.  State names match the MAST taxonomy.
    """
    state_names = [
        "plan",       # 0 — agent forms a plan
        "tool_call",  # 1 — agent invokes an external tool
        "observe",    # 2 — agent reads tool output
        "reflect",    # 3 — agent self-critiques
        "answer",     # 4 — agent produces final answer (before grading)
    ]
    m = len(state_names)

    # Realistic forward-leaning transition structure
    Q = np.array([
        [0.00, 0.70, 0.00, 0.10, 0.10],  # plan -> tool_call / reflect / answer often
        [0.00, 0.10, 0.70, 0.00, 0.00],  # tool_call -> observe
        [0.00, 0.20, 0.00, 0.40, 0.20],  # observe -> tool_call / reflect / answer
        [0.10, 0.20, 0.00, 0.00, 0.40],  # reflect -> plan / tool_call / answer
        [0.00, 0.00, 0.00, 0.10, 0.00],  # answer -> reflect (retry)
    ])
    R_succ = np.array([0.00, 0.00, 0.05, 0.05, 0.40])
    R_fail = np.array([0.10, 0.20, 0.05, 0.10, 0.10])
    # Ensure rows sum to 1
    for i in range(m):
        total = Q[i].sum() + R_succ[i] + R_fail[i]
        if abs(total - 1.0) > 1e-8:
            deficit = 1.0 - total
            R_fail[i] += deficit   # absorb any rounding deficit into failure

    s0 = state_names.index("plan")
    return Q, R_succ, R_fail, s0, state_names
