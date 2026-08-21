"""SS13 -- Bootstrap uncertainty: fixed-clustering vs full re-cluster.

Addresses R4 uncertainty question: "clarify whether the reported bootstrap
intervals re-run the full clustering or use the faster path that holds
clustering fixed, since the latter can understate uncertainty."

The reported SS8 intervals use the fast path (nearest-centroid reassignment to
the target clustering). The module also implements a full re-cluster bootstrap
(``fast=False``), which re-runs Ward clustering on every resample and therefore
additionally captures clustering instability. This script quantifies the gap by
running BOTH paths on a controlled corpus and comparing the median per-entry CI
half-width of the transition matrix.

Run:  python SS13_bootstrap_paths.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "sim"))

from mcr.trace_to_chain import TraceStep, fit
from mcr.uncertainty import bootstrap_intervals

SEED = 20260606
DATA = ROOT / "data" / "swebench_real"
N_TRACES = 250
N_BOOT = 80


def gt():
    Q = np.array([
        [0.10, 0.65, 0.05, 0.05],
        [0.05, 0.10, 0.65, 0.05],
        [0.15, 0.10, 0.10, 0.35],
        [0.20, 0.05, 0.10, 0.10],
    ])
    Rs = np.array([0.10, 0.10, 0.20, 0.45]); Rf = np.array([0.05, 0.05, 0.05, 0.05])
    for i in range(4):
        t = Q[i].sum() + Rs[i] + Rf[i]; Q[i] /= t; Rs[i] /= t; Rf[i] /= t
    return Q, Rs, Rf


def sample(Q, Rs, Rf, rng, dmax=80):
    m = Q.shape[0]; cum = np.cumsum(np.concatenate([Q, Rs[:, None], Rf[:, None]], 1), 1)
    s = 0; seq = []
    for _ in range(dmax):
        seq.append(s); u = rng.random(); nx = int(np.searchsorted(cum[s], u))
        if nx == m: return seq, "success"
        if nx == m + 1: return seq, "failure"
        s = nx
    return seq, "failure"


def oh(s, m):
    v = np.zeros(m); v[s] = 1.0; return v


def main():
    Q, Rs, Rf = gt(); m = Q.shape[0]
    rng = np.random.default_rng(SEED)
    traces = []
    for _ in range(N_TRACES):
        seq, term = sample(Q, Rs, Rf, rng)
        steps = [TraceStep(features=oh(s, m)) for s in seq]
        steps.append(TraceStep(features=np.zeros(m), is_terminal=True, terminal_label=term))
        traces.append(steps)
    target = fit(traces, k_max=6)
    print(f"SS13 bootstrap paths: N={N_TRACES} traces, fitted m={target.m}, n_boot={N_BOOT}")

    def med_halfwidth(ci):
        # half-width of Q CIs, averaged over entries
        w = (ci.Q_hi - ci.Q_lo) / 2.0
        return float(np.median(w))

    fast = bootstrap_intervals(traces, n_boot=N_BOOT, seed=SEED + 1,
                               target_fit=target, fast=True)
    full = bootstrap_intervals(traces, n_boot=N_BOOT, seed=SEED + 1,
                               target_fit=target, fast=False)
    hf_fast, hf_full = med_halfwidth(fast), med_halfwidth(full)
    ratio = hf_full / hf_fast if hf_fast > 0 else float("nan")
    print(f"  median Q CI half-width  fast (fixed clustering) = {hf_fast:.4f}")
    print(f"  median Q CI half-width  full (re-cluster)       = {hf_full:.4f}")
    print(f"  ratio full/fast = {ratio:.2f}x  "
          f"(full re-cluster intervals are {'wider' if ratio>1 else 'not wider'})")
    summary = dict(N=N_TRACES, m=target.m, n_boot=N_BOOT,
                   median_halfwidth_fast=round(hf_fast, 4),
                   median_halfwidth_full=round(hf_full, 4),
                   ratio_full_over_fast=round(ratio, 3))
    (DATA / "SS13_bootstrap_paths.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {DATA/'SS13_bootstrap_paths.json'}")


if __name__ == "__main__":
    main()
