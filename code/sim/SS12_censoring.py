"""SS12 -- Censoring-sensitivity of the exit estimates.

Addresses R4 GoF-Q2: "the censoring discussion reports under 10% censoring on
synthetic data, but real traces may be censored far more heavily, which would
bias the exit estimates."

We take a KNOWN ground-truth absorbing chain (so the true R_infinity is exact),
sample trajectories, then right-censor a controlled fraction of them (truncate
at a uniform-random transient step and mark the trace as censored, i.e.,
terminal label is neither success nor failure). We refit with TraceToChain at
each censoring level and report the bias in the recovered R_infinity relative
to the truth. This quantifies how heavy censoring biases the exit estimates,
and at what level the bias becomes material.

Run:  python SS12_censoring.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "sim"))

from mcr.trace_to_chain import TraceStep, fit
from mcr.reliability import asymptotic_reliability

SEED = 20260606
DATA = ROOT / "data" / "swebench_real"  # reuse output dir
N_TRACES = 600
CENSOR_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]


def ground_truth():
    """A 5-state absorbing chain (plan->tool->observe->reflect->answer-ish)."""
    m = 5
    Q = np.array([
        [0.05, 0.70, 0.00, 0.05, 0.00],
        [0.00, 0.05, 0.75, 0.05, 0.00],
        [0.10, 0.10, 0.05, 0.40, 0.20],
        [0.20, 0.05, 0.05, 0.05, 0.40],
        [0.05, 0.05, 0.05, 0.05, 0.10],
    ])
    R_succ = np.array([0.15, 0.10, 0.10, 0.20, 0.60])
    R_fail = np.array([0.05, 0.05, 0.05, 0.05, 0.05])
    # normalize rows to sum to 1 (Q row + succ + fail)
    for i in range(m):
        tot = Q[i].sum() + R_succ[i] + R_fail[i]
        Q[i] /= tot; R_succ[i] /= tot; R_fail[i] /= tot
    return Q, R_succ, R_fail


def sample_traj(Q, Rs, Rf, s0, rng, d_max=200):
    """Return (state_seq, terminal) where terminal in {'success','failure'}."""
    m = Q.shape[0]
    full = np.concatenate([Q, Rs[:, None], Rf[:, None]], axis=1)
    cum = np.cumsum(full, axis=1)
    s = s0; seq = []
    for _ in range(d_max):
        seq.append(s)
        u = rng.random(); nxt = int(np.searchsorted(cum[s], u))
        if nxt == m:
            return seq, "success"
        if nxt == m + 1:
            return seq, "failure"
        s = nxt
    return seq, "failure"


def onehot(state, m):
    v = np.zeros(m); v[state] = 1.0; return v


def build_traces(seqs_terms, m, censor_frac, rng):
    traces = []
    n = len(seqs_terms)
    cens_idx = set(rng.choice(n, size=int(round(censor_frac * n)), replace=False).tolist()) \
        if censor_frac > 0 else set()
    for i, (seq, term) in enumerate(seqs_terms):
        if i in cens_idx and len(seq) >= 2:
            cut = int(rng.integers(1, len(seq)))  # keep 1..len-1 transient steps
            steps = [TraceStep(features=onehot(s, m)) for s in seq[:cut]]
            steps.append(TraceStep(features=np.zeros(m), is_terminal=True, terminal_label=None))
        else:
            steps = [TraceStep(features=onehot(s, m)) for s in seq]
            steps.append(TraceStep(features=np.zeros(m), is_terminal=True, terminal_label=term))
        traces.append(steps)
    return traces


def main():
    Q, Rs, Rf = ground_truth()
    m = Q.shape[0]
    true_Rinf = float(asymptotic_reliability(Q, Rs, s0=0))
    rng = np.random.default_rng(SEED)
    base = [sample_traj(Q, Rs, Rf, 0, rng) for _ in range(N_TRACES)]
    n_succ = sum(1 for _, t in base if t == "success")
    print(f"SS12 censoring sensitivity: true R_inf={true_Rinf:.4f}; "
          f"N={N_TRACES} traces (success={n_succ}); m={m}")
    print(f"{'censor%':>7} | {'m_hat':>5} {'Rinf_hat':>8} {'abs_bias':>8} {'rel_bias%':>9}")
    rows = []
    for c in CENSOR_LEVELS:
        rng2 = np.random.default_rng(SEED + 7)
        traces = build_traces(base, m, c, rng2)
        chain = fit(traces, k_max=8)
        # align: chain states are clusters of one-hot vectors -> recovers m states;
        # use the fitted chain's own s0_cluster for R_inf.
        rinf = float(asymptotic_reliability(chain.Q, chain.R_succ, s0=chain.s0_cluster))
        bias = rinf - true_Rinf
        rows.append(dict(censor=c, m=chain.m, Rinf_hat=round(rinf, 4),
                         abs_bias=round(bias, 4), rel_bias_pct=round(100 * bias / true_Rinf, 2)))
        print(f"{100*c:7.0f} | {chain.m:>5} {rinf:8.4f} {bias:+8.4f} {100*bias/true_Rinf:+9.2f}")
    summary = dict(true_Rinf=true_Rinf, N=N_TRACES, m_true=m, rows=rows)
    (DATA / "SS12_censoring.json").write_text(json.dumps(summary, indent=2))
    write_tex(summary, DATA / "SS12_censoring.tex")
    # headline: censoring level where |rel bias| first exceeds 5%
    over = [r["censor"] for r in rows if abs(r["rel_bias_pct"]) > 5]
    print(f"\n|relative bias|>5% first at censoring = "
          f"{(min(over) if over else 'never in range')}")
    print(f"wrote {DATA/'SS12_censoring.json'} and SS12_censoring.tex")


def write_tex(summary, path):
    L = ["% Auto-generated by SS12_censoring.py",
         "\\begin{table}[!t]\\centering",
         "\\caption{SS12: censoring sensitivity on a ground-truth chain ($R_\\infty=0.784$).}",
         "\\label{tab:ss12}",
         "\\begin{tabular}{rrrr}", "\\toprule",
         "censoring & $\\hat m$ & $\\hat R_\\infty$ & rel.\\ bias \\\\", "\\midrule"]
    for r in summary["rows"]:
        L.append(f"{100*r['censor']:.0f}\\% & {r['m']} & {r['Rinf_hat']:.3f} & "
                 f"{r['rel_bias_pct']:+.1f}\\% \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
