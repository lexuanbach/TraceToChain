"""SS11 -- Clustering / featurization sensitivity ablation.

Addresses R2 ("no ablation or sensitivity study on clustering/featurization")
and R4 GoF-Q1 ("the GoF verdict is a joint test of featurization + clustering
+ first-order Markov; its sensitivity to the clustering choice -- linkage,
silhouette, k range -- should be characterized").

We vary, on the real SWE-bench corpus (SS10), three knobs and re-run the full
fit + composite goodness-of-fit:
  * featurization: action-category one-hot only  vs  category + step-depth;
  * clustering linkage: ward / average / complete;
  * the k search range cap k_max in {6, 8, 10}.
For each configuration we report the recovered state count m, silhouette,
Delta_AIC (1st-2nd), held-out KS p, L_inf RDC, and the composite verdict.

The purpose is to characterize how much the *estimates* (m, silhouette) move
with these choices and whether the *verdict* is stable.

Run:  python SS11_ablation.py --limit 160
"""
from __future__ import annotations
import argparse, json, sys, itertools
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code")); sys.path.insert(0, str(ROOT / "code" / "sim"))

from mcr.trace_to_chain import fit, goodness_of_fit, empirical_first_passage_from_traces
from mcr.reliability import reliability_curve, asymptotic_reliability
from swebench_adapter import load_corpus

SEED = 20260606
D_MAX = 80
DATA = ROOT / "data" / "swebench_real"


def empirical_rdc(traces, d_grid):
    n = len(traces); succ = [sum(1 for s in t if not s.is_terminal)
                             for t in traces if t[-1].terminal_label == "success"]
    succ = np.asarray(succ, float)
    return np.array([(np.sum(succ <= d)/n if n else 0.0) for d in d_grid])


def split(traces, seed):
    rng = np.random.default_rng(seed); idx = np.arange(len(traces)); rng.shuffle(idx)
    h = len(idx)//2
    return [traces[i] for i in idx[:h]], [traces[i] for i in idx[h:]]


def run_cfg(corpus_path, depth, linkage, kmax, limit):
    traces, ids, stats = load_corpus(corpus_path, include_depth=depth)
    if limit and len(traces) > limit:
        rng0 = np.random.default_rng(SEED)
        sel = np.sort(rng0.choice(len(traces), size=limit, replace=False))
        traces = [traces[i] for i in sel]
    fit_tr, test_tr = split(traces, SEED)
    chain = fit(fit_tr, k_max=kmax, linkage=linkage)
    emp = empirical_first_passage_from_traces(
        [[0 for s in t if not s.is_terminal] for t in test_tr],
        [t[-1].terminal_label for t in test_tr])
    gof = goodness_of_fit(chain, emp, n_model_samples=15000,
                          s0=chain.s0_cluster, rng=np.random.default_rng(SEED+1))
    d = np.arange(0, D_MAX+1)
    Ra = np.asarray(reliability_curve(chain.Q, chain.R_succ, s0=chain.s0_cluster, d_max=D_MAX))
    Re = empirical_rdc(test_tr, d)
    Linf = float(np.max(np.abs(Ra-Re)))
    Rinf = float(asymptotic_reliability(chain.Q, chain.R_succ, s0=chain.s0_cluster))
    composite = (gof["ks_pvalue"] > 0.05) and bool(chain.first_order_preferred)
    return dict(depth=depth, linkage=linkage, kmax=kmax, m=chain.m,
                silhouette=round(chain.silhouette, 3),
                delta_aic=round(chain.aic_1st - chain.aic_2nd, 1),
                first_order=bool(chain.first_order_preferred),
                ks_p=round(gof["ks_pvalue"], 4), L_inf=round(Linf, 3),
                R_inf=round(Rinf, 3), verdict=("ACCEPT" if composite else "REJECT"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(DATA / "corpus.json"))
    ap.add_argument("--limit", type=int, default=160)
    args = ap.parse_args()
    rows = []
    grid = list(itertools.product([True, False], ["ward", "average", "complete"], [6, 8, 10]))
    print(f"SS11 ablation: {len(grid)} configurations on {args.corpus} (limit={args.limit})")
    print(f"{'depth':5} {'linkage':8} {'kmax':4} | {'m':>2} {'silh':>5} {'dAIC':>7} "
          f"{'1st?':>4} {'KSp':>6} {'Linf':>5} {'Rinf':>5} | verdict")
    for depth, link, kmax in grid:
        try:
            r = run_cfg(args.corpus, depth, link, kmax, args.limit)
        except Exception as e:
            print(f"  {str(depth):5} {link:8} {kmax:4} | ERROR {str(e)[:50]}"); continue
        rows.append(r)
        print(f"{str(r['depth']):5} {r['linkage']:8} {r['kmax']:<4} | {r['m']:>2} "
              f"{r['silhouette']:>5} {r['delta_aic']:>7} {str(r['first_order']):>5} "
              f"{r['ks_p']:>6} {r['L_inf']:>5} {r['R_inf']:>5} | {r['verdict']}")
    n_reject = sum(1 for r in rows if r["verdict"] == "REJECT")
    ms = [r["m"] for r in rows]
    summary = dict(n_cfg=len(rows), n_reject=n_reject, n_accept=len(rows)-n_reject,
                   m_min=min(ms), m_max=max(ms), rows=rows)
    print(f"\nverdict REJECT in {n_reject}/{len(rows)} configurations; "
          f"recovered m ranges {min(ms)}-{max(ms)}.")
    (DATA / "SS11_ablation.json").write_text(json.dumps(summary, indent=2))
    write_tex(summary, DATA / "SS11_ablation.tex")
    print(f"wrote {DATA/'SS11_ablation.json'} and SS11_ablation.tex")


def write_tex(summary, path):
    L = ["% Auto-generated by SS11_ablation.py",
         "\\begin{table}[!t]\\centering",
         "\\caption{SS11: clustering and featurization ablation ($18$ configurations).}",
         "\\label{tab:ss11}\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{llrrrrrl}", "\\toprule",
         "feat. & linkage & $k_{\\max}$ & $m$ & silh. & $\\Delta_{\\mathrm{AIC}}$ & KS $p$ & verdict \\\\",
         "\\midrule"]
    for r in summary["rows"]:
        feat = "cat+depth" if r["depth"] else "cat"
        L.append(f"{feat} & {r['linkage']} & {r['kmax']} & {r['m']} & {r['silhouette']:.2f} & "
                 f"{r['delta_aic']:+.0f} & {r['ks_p']:.3f} & {r['verdict']} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
