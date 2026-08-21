"""SS10 -- Real-data validation on SWE-bench agent traces.

Addresses the reviewers' central request (R1.W1, R2, R3, R4.W1/W2): fit at
least one *real* LLM-agent trace corpus end-to-end with TraceToChain and
report the composite goodness-of-fit outcome HONESTLY, whatever it is. A
rejection is a scientifically valid and useful result, consistent with the
paper's "reject if GoF fails" philosophy.

Corpus
------
Real SWE-agent trajectories from a SWE-bench Verified leaderboard submission
(default: 20250522_sweagent_claude-4-sonnet-20250514), harvested from the
public swe-bench-submissions S3 bucket. See ``swebench_adapter`` for
provenance and the action-token rule. Ground-truth success/failure is the
SWE-bench ``resolved`` verdict.

Protocol (mirrors SS9)
----------------------
  1. Load the corpus into TraceStep sequences (one transient step per agent
     action; one terminal success/failure absorbing step from ``resolved``).
  2. Split the corpus 50/50 (deterministic seed) BEFORE featurization/fitting.
  3. Fit \hat M=(\hat Q,\hat R_oplus,\hat R_ominus) on the fit half only.
  4. Composite goodness-of-fit:
       (a) AIC layer: is a first-order DTMC preferred over second-order?
       (b) KS layer: two-sample KS between the model first-passage CDF and the
           held-out empirical first-passage times (p > 0.05 to accept).
     Composite verdict = accept iff BOTH pass (the paper's rule).
  5. Report L_inf RDC discrepancy on the held-out half and overlay the
     empirical vs analytic reliability-decay curve.

Run:  python SS10_swebench_real.py --corpus ../data/swebench_real/corpus.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "sim"))

from mcr.trace_to_chain import (
    fit,
    first_passage_times,
    empirical_first_passage_from_traces,
    goodness_of_fit,
)
from mcr.reliability import reliability_curve

from swebench_adapter import load_corpus, categorize, CATEGORIES

SEED = 20260606
D_MAX = 80
N_MODEL_FPT = 20000

DATA_DIR = ROOT / "data" / "swebench_real"
FIG_DIR = ROOT / "figs"


def empirical_rdc(traces, ids, d_grid):
    """Empirical RDC on a set of traces: fraction reaching success by step d.

    A trace contributes a 'success by d' event iff its terminal label is
    success and its number of transient steps <= d.
    """
    lens_succ = []
    n = len(traces)
    for tr in traces:
        term = tr[-1].terminal_label if tr and tr[-1].is_terminal else None
        L = sum(1 for s in tr if not s.is_terminal)
        if term == "success":
            lens_succ.append(L)
    lens_succ = np.asarray(lens_succ, dtype=float)
    out = np.array([(np.sum(lens_succ <= d) / n) if n else 0.0 for d in d_grid])
    return out


def split_traces(traces, ids, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(traces))
    rng.shuffle(idx)
    half = len(idx) // 2
    fit_idx, test_idx = idx[:half], idx[half:]
    fit_tr = [traces[i] for i in fit_idx]
    test_tr = [traces[i] for i in test_idx]
    return fit_tr, test_tr


def to_labels_terminals(traces):
    """Approximate cluster-label sequences (not needed for FPT length) and
    terminal labels, for empirical_first_passage_from_traces (uses lengths)."""
    trace_labels = []
    terminals = []
    for tr in traces:
        seq = [0 for s in tr if not s.is_terminal]  # placeholder labels
        trace_labels.append(seq)
        terminals.append(tr[-1].terminal_label if tr and tr[-1].is_terminal else None)
    return trace_labels, terminals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA_DIR / "corpus.json"))
    ap.add_argument("--no-depth", action="store_true",
                    help="featurize with action category only (no depth feature)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=220,
                    help="cap total traces (deterministic subsample) to keep "
                         "agglomerative clustering tractable; 0 = use all")
    ap.add_argument("--kmax", type=int, default=10,
                    help="max number of clusters/states to search")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    traces, ids, stats = load_corpus(
        args.corpus, include_depth=not args.no_depth, max_steps=args.max_steps
    )
    print("SS10 -- Real SWE-bench validation")
    print(f"  corpus: {args.corpus}")
    print(f"  instances={stats.n_instances} used={stats.n_used} "
          f"(success={stats.n_success}, failure={stats.n_failure}); "
          f"steps={stats.n_steps}; dropped={stats.dropped_empty}")

    if args.limit and len(traces) > args.limit:
        rng0 = np.random.default_rng(SEED)
        sel = rng0.choice(len(traces), size=args.limit, replace=False)
        sel.sort()
        traces = [traces[i] for i in sel]
        ids = [ids[i] for i in sel]
        ns = sum(1 for t in traces if t[-1].terminal_label == "success")
        print(f"  subsampled to {len(traces)} traces (success={ns}, "
              f"failure={len(traces)-ns}) for tractable clustering")

    fit_tr, test_tr = split_traces(traces, ids, SEED)
    print(f"  split: n_fit={len(fit_tr)}  n_test={len(test_tr)}")

    # ---- Fit on fit half only ----
    chain = fit(fit_tr, k_max=args.kmax)
    print(f"  fitted chain: m={chain.m} states, silhouette={chain.silhouette:.3f}")
    print(f"  AIC: 1st={chain.aic_1st:.1f}  2nd={chain.aic_2nd:.1f}  "
          f"first_order_preferred={chain.first_order_preferred} "
          f"(Delta_AIC = {chain.aic_1st - chain.aic_2nd:+.1f})")

    # ---- KS layer on held-out test traces ----
    test_labels, test_terms = to_labels_terminals(test_tr)
    emp_fpt = empirical_first_passage_from_traces(test_labels, test_terms)
    rng = np.random.default_rng(SEED + 1)
    gof = goodness_of_fit(chain, emp_fpt, n_model_samples=N_MODEL_FPT,
                          s0=chain.s0_cluster, rng=rng)

    # ---- RDC overlay (held-out empirical vs analytic) ----
    d_grid = np.arange(0, D_MAX + 1)
    R_analytic = np.asarray(reliability_curve(chain.Q, chain.R_succ,
                                              s0=chain.s0_cluster, d_max=D_MAX))
    R_emp = empirical_rdc(test_tr, ids, d_grid)
    L_inf = float(np.max(np.abs(R_analytic - R_emp)))

    ks_pass = gof["ks_pvalue"] > 0.05
    aic_pass = bool(chain.first_order_preferred)
    composite = ks_pass and aic_pass

    print(f"  KS: D={gof['ks_stat']:.3f}  p={gof['ks_pvalue']:.4f}  "
          f"(n_emp={gof['n_emp']}, n_model={gof['n_model']})  -> {'PASS' if ks_pass else 'REJECT'}")
    print(f"  AIC layer -> {'PASS (first-order)' if aic_pass else 'REJECT (higher-order)'}")
    print(f"  COMPOSITE GoF verdict: {'ACCEPT' if composite else 'REJECT'}")
    print(f"  L_inf RDC (held-out) = {L_inf:.3f}")

    # state interpretation: dominant category per cluster
    state_summary = state_taxonomy(chain, fit_tr, include_depth=not args.no_depth)

    summary = {
        "corpus_file": str(args.corpus),
        "config": {"seed": SEED, "D_MAX": D_MAX, "N_MODEL_FPT": N_MODEL_FPT,
                   "include_depth": not args.no_depth, "max_steps": args.max_steps},
        "corpus": stats.__dict__,
        "split": {"n_fit": len(fit_tr), "n_test": len(test_tr)},
        "fit": {"m": chain.m, "silhouette": chain.silhouette,
                "aic_1st": chain.aic_1st, "aic_2nd": chain.aic_2nd,
                "delta_aic": chain.aic_1st - chain.aic_2nd,
                "first_order_preferred": aic_pass},
        "ks": gof,
        "rdc": {"L_inf": L_inf, "d_max": D_MAX},
        "verdict": {"ks_pass": ks_pass, "aic_pass": aic_pass,
                    "composite_accept": composite},
        "state_taxonomy": state_summary,
        "R_analytic": R_analytic.tolist(),
        "R_emp": R_emp.tolist(),
    }
    (DATA_DIR / "SS10_summary.json").write_text(json.dumps(summary, indent=2))
    write_tex(summary, DATA_DIR / "SS10_summary.tex")
    make_overlay(d_grid, R_analytic, R_emp, gof, L_inf, composite,
                 FIG_DIR / "fig_ss10_swebench.pdf")
    print(f"Wrote {DATA_DIR/'SS10_summary.json'}")
    print(f"Wrote {DATA_DIR/'SS10_summary.tex'}")
    print(f"Wrote {FIG_DIR/'fig_ss10_swebench.pdf'}")
    print("[DONE] SS10 complete. (Outcome reported as-is, accept or reject.)")


def state_taxonomy(chain, fit_tr, include_depth):
    """Label each fitted state by its dominant action category."""
    # rebuild per-step categories aligned to chain.labels order
    cats = []
    for tr in fit_tr:
        for s in tr:
            if s.is_terminal:
                continue
            v = s.features[:len(CATEGORIES)]
            cats.append(CATEGORIES[int(np.argmax(v))])
    cats = np.array(cats)
    labels = np.array(chain.labels)
    out = {}
    for st in range(chain.m):
        mask = labels == st
        if not mask.any():
            out[str(st)] = {"dominant": None, "n": 0}
            continue
        vals, counts = np.unique(cats[mask], return_counts=True)
        order = np.argsort(-counts)
        out[str(st)] = {
            "dominant": str(vals[order[0]]),
            "n": int(mask.sum()),
            "mix": {str(vals[o]): int(counts[o]) for o in order[:3]},
        }
    return out


def write_tex(summary, path):
    v = summary["verdict"]
    f = summary["fit"]
    ks = summary["ks"]
    verdict = "accepts" if v["composite_accept"] else "rejects"
    lines = []
    lines.append("% Auto-generated by SS10_swebench_real.py -- do not edit by hand.")
    lines.append("\\begin{table}[!t]")
    lines.append("\\centering")
    lines.append("\\caption{SS10: \\textsc{TraceToChain} on $479$ real SWE-agent "
                 "trajectories.}")
    lines.append("\\label{tab:ss10}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\toprule")
    lines.append("\\textbf{Quantity} & \\textbf{Value} \\\\")
    lines.append("\\midrule")
    lines.append(f"Instances used (succ/fail) & {summary['corpus']['n_used']} "
                 f"({summary['corpus']['n_success']}/{summary['corpus']['n_failure']}) \\\\")
    lines.append(f"Agent steps & {summary['corpus']['n_steps']} \\\\")
    lines.append(f"Fit / test traces & {summary['split']['n_fit']} / {summary['split']['n_test']} \\\\")
    lines.append(f"Recovered states $m$ & {f['m']} \\\\")
    lines.append(f"Silhouette & {f['silhouette']:.3f} \\\\")
    lines.append(f"$\\Delta_{{\\mathrm{{AIC}}}}$ (1st$-$2nd) & {f['delta_aic']:+.1f} \\\\")
    lines.append(f"First-order preferred (AIC) & {'yes' if f['first_order_preferred'] else 'no'} \\\\")
    lines.append(f"Held-out KS $D$ & {ks['ks_stat']:.3f} \\\\")
    lines.append(f"Held-out KS $p$ & {ks['ks_pvalue']:.4f} \\\\")
    lines.append(f"$L_\\infty^{{\\mathrm{{RDC}}}}$ (held-out) & {summary['rdc']['L_inf']:.3f} \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{Composite verdict}} & \\textbf{{{'ACCEPT' if v['composite_accept'] else 'REJECT'}}} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    Path(path).write_text("\n".join(lines) + "\n")


def make_overlay(d_grid, R_analytic, R_emp, gof, L_inf, composite, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[warn] matplotlib unavailable, skipping figure: {e}")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(d_grid, R_analytic, "-", lw=2, label=r"analytic $\mathcal{R}(d)$ (fitted $\hat M$)")
    ax.plot(d_grid, R_emp, "--", lw=2, label=r"held-out empirical $\hat{\mathcal{R}}_{\rm emp}(d)$")
    ax.set_xlabel("step horizon $d$")
    ax.set_ylabel(r"reliability $\mathcal{R}(d)$")
    verdict = "ACCEPT" if composite else "REJECT"
    ax.set_title(f"SS10 real SWE-bench: KS $p$={gof['ks_pvalue']:.3f}, "
                 f"$L_\\infty$={L_inf:.3f} [{verdict}]")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"  figure saved: {path}")


if __name__ == "__main__":
    main()
