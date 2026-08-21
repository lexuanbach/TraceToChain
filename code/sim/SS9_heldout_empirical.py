"""SS9 -- Held-out empirical validation of the TraceToChain pipeline.

Addresses reviewer W11: the SS7(C) goodness-of-fit check previously drew its
``empirical'' first-passage times from synthetic traces sampled from the
fitted chain, which is circular.  SS9 instead performs a strict
fit/test split on the synthetic-MAST corpus:

  1.  For each of 7 MAST-style frameworks, generate a trace corpus and
      split 50/50 (fit_traces, test_traces) with a deterministic seed.
  2.  Fit \hat M = (\hat Q, \hat R_\oplus, \hat R_\ominus) on fit_traces
      ONLY, obtaining a clustered state space and transition matrix.
  3.  From \hat M compute the analytic reliability-decay curve
      \mathcal R(d) = \Pr_{\hat M}[\tau_\oplus \le d] and the analytic
      first-passage CDF F_{\tau_\oplus}^{\hat M}.
  4.  On the *held-out* test traces compute the EMPIRICAL counterparts
      \hat{\mathcal R}_{\mathrm{emp}}(d) (success-by-d rate) and the
      empirical first-passage CDF \hat F_{\tau_\oplus}.
  5.  Report
        D_{\mathrm{KS}}, p_{\mathrm{KS}} on \{F^{\hat M}_\tau, \hat F_\tau\}
        L_\infty^{\mathrm{RDC}} = \sup_d |\mathcal R(d) - \hat{\mathcal R}_{\mathrm{emp}}(d)|
      per framework, in data/mast_derived/SS9_heldout_summary.{json,tex},
      plus an overlay figure figs/fig_rdc_overlay.pdf.

The split is performed BEFORE clustering/fitting, so the held-out test
traces are never seen by either the featurizer or the MLE.

Usage
-----
    python SS9_heldout_empirical.py
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
    TraceStep,
    fit,
    first_passage_times,
)
from mcr.reliability import reliability_curve


# Framework-specific ground-truth absorbing Markov chains.  Each entry
# is (Q*, R_oplus*, R_ominus*, state_names) with m* transient states
# that loosely mimic the MAST taxonomy (plan, tool, observe, reflect,
# answer, retry).  Rows obey Q* 1 + R_oplus* + R_ominus* = 1.
def _gt_chain(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return a ground-truth absorbing chain for one MAST framework."""
    tables = {
        # react: plan -> tool_call -> observe loop, occasional reflect
        "react": (
            [[0.00, 0.75, 0.00, 0.05, 0.10],
             [0.00, 0.10, 0.72, 0.00, 0.00],
             [0.00, 0.25, 0.00, 0.30, 0.30],
             [0.15, 0.30, 0.00, 0.00, 0.35],
             [0.00, 0.00, 0.00, 0.15, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.65],
            [0.10, 0.18, 0.10, 0.20, 0.20],
            ["plan", "tool_call", "observe", "reflect", "answer"],
        ),
        # reflexion: heavy reflect-answer loop
        "reflexion": (
            [[0.00, 0.60, 0.00, 0.15, 0.15],
             [0.00, 0.15, 0.68, 0.00, 0.00],
             [0.00, 0.20, 0.00, 0.40, 0.25],
             [0.05, 0.25, 0.00, 0.10, 0.45],
             [0.00, 0.00, 0.00, 0.25, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.60],
            [0.10, 0.17, 0.15, 0.15, 0.15],
            ["plan", "tool_call", "observe", "reflect", "answer"],
        ),
        # cot_agent: plan-heavy, few tool calls, direct answer
        "cot_agent": (
            [[0.20, 0.20, 0.00, 0.30, 0.20],
             [0.00, 0.20, 0.60, 0.00, 0.00],
             [0.00, 0.30, 0.00, 0.20, 0.35],
             [0.25, 0.10, 0.00, 0.10, 0.40],
             [0.00, 0.00, 0.00, 0.20, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.55],
            [0.10, 0.20, 0.15, 0.15, 0.25],
            ["plan", "tool_call", "observe", "reflect", "answer"],
        ),
        # toolformer: tool-call dominant
        "toolformer": (
            [[0.00, 0.80, 0.00, 0.05, 0.05],
             [0.00, 0.05, 0.80, 0.00, 0.00],
             [0.00, 0.35, 0.00, 0.15, 0.35],
             [0.10, 0.40, 0.00, 0.10, 0.30],
             [0.00, 0.00, 0.00, 0.10, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.70],
            [0.10, 0.15, 0.15, 0.10, 0.20],
            ["plan", "tool_call", "observe", "reflect", "answer"],
        ),
        # babyagi: longer chains, frequent retry
        "babyagi": (
            [[0.00, 0.50, 0.00, 0.10, 0.10, 0.20],
             [0.00, 0.10, 0.65, 0.00, 0.00, 0.10],
             [0.00, 0.20, 0.00, 0.30, 0.20, 0.15],
             [0.10, 0.30, 0.00, 0.10, 0.25, 0.15],
             [0.00, 0.00, 0.00, 0.20, 0.00, 0.15],
             [0.20, 0.30, 0.00, 0.20, 0.15, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.50, 0.00],
            [0.20, 0.15, 0.15, 0.10, 0.15, 0.15],
            ["plan", "tool_call", "observe", "reflect", "answer", "retry"],
        ),
        # autogpt: long plan-tool-observe with frequent retry
        "autogpt": (
            [[0.10, 0.55, 0.00, 0.05, 0.05, 0.15],
             [0.00, 0.10, 0.70, 0.00, 0.00, 0.10],
             [0.00, 0.25, 0.00, 0.25, 0.25, 0.15],
             [0.10, 0.30, 0.00, 0.10, 0.30, 0.10],
             [0.00, 0.00, 0.00, 0.15, 0.00, 0.15],
             [0.20, 0.35, 0.00, 0.15, 0.10, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.55, 0.00],
            [0.15, 0.10, 0.10, 0.05, 0.15, 0.20],
            ["plan", "tool_call", "observe", "reflect", "answer", "retry"],
        ),
        # agentbench: balanced
        "agentbench": (
            [[0.05, 0.50, 0.00, 0.15, 0.15],
             [0.00, 0.15, 0.65, 0.00, 0.00],
             [0.00, 0.25, 0.00, 0.30, 0.30],
             [0.10, 0.25, 0.00, 0.10, 0.40],
             [0.00, 0.00, 0.00, 0.15, 0.00]],
            [0.00, 0.00, 0.05, 0.05, 0.60],
            [0.15, 0.20, 0.15, 0.15, 0.25],
            ["plan", "tool_call", "observe", "reflect", "answer"],
        ),
    }
    Q_list, Rs_list, Rf_list, names = tables[name]
    Q = np.asarray(Q_list, dtype=float)
    Rs = np.asarray(Rs_list, dtype=float)
    Rf = np.asarray(Rf_list, dtype=float)
    # Ensure row normalization (absorb any rounding into R_fail).
    for i in range(Q.shape[0]):
        tot = Q[i].sum() + Rs[i] + Rf[i]
        Rf[i] += 1.0 - tot
    return Q, Rs, Rf, names


def make_markov_mast_corpus(
    name: str, seed: int, n_traces: int,
    noise_scale: float = 0.08,
    censor_prob: float = 0.05,
) -> list[list[TraceStep]]:
    """Sample ``n_traces`` traces from the ground-truth chain for framework
    ``name``.  Each latent state emits a noisy one-hot feature vector
    (noise ~ N(0, noise_scale^2)) so that the TraceToChain clustering
    has a nontrivial job.  With probability ``censor_prob``, a trace
    is truncated at a random step and marked as censored (terminal_label=None)
    so that SS9 exercises the variable-horizon path.
    """
    rng = np.random.default_rng(seed)
    Q, Rs, Rf, names = _gt_chain(name)
    m = Q.shape[0]
    # Initial distribution: draw s0 from a simple prior (skewed toward
    # state 0 = plan, mimicking MAST trace starts).
    pi0 = np.zeros(m)
    pi0[0] = 0.7
    if m >= 2:
        pi0[1] = 0.15
    for j in range(2, m):
        pi0[j] = 0.15 / max(1, m - 2)
    pi0 = pi0 / pi0.sum()

    # Emission: noisy one-hot over m dimensions.
    def emit(s: int) -> np.ndarray:
        x = np.zeros(m)
        x[s] = 1.0
        x += rng.normal(scale=noise_scale, size=m)
        return x

    traces: list[list[TraceStep]] = []
    full = np.concatenate([Q, Rs[:, None], Rf[:, None]], axis=1)
    cum = np.cumsum(full, axis=1)

    for _ in range(n_traces):
        s = int(rng.choice(m, p=pi0))
        steps: list[TraceStep] = []
        censor_at = None
        if rng.random() < censor_prob:
            censor_at = int(rng.integers(2, 10))  # truncate at step 2..9
        terminal_label: str | None = None
        for t in range(1, 500):
            steps.append(TraceStep(features=emit(s)))
            if censor_at is not None and t >= censor_at:
                terminal_label = None
                break
            u = rng.random()
            nxt = int(np.searchsorted(cum[s], u))
            if nxt == m:
                terminal_label = "success"
                break
            if nxt == m + 1:
                terminal_label = "failure"
                break
            s = nxt
        steps.append(TraceStep(
            features=np.zeros(m),
            is_terminal=True,
            terminal_label=terminal_label,
        ))
        traces.append(steps)
    return traces


def first_passage_times_pi0(
    Q: np.ndarray,
    R_succ: np.ndarray,
    R_fail: np.ndarray,
    pi0: np.ndarray,
    n_samples: int,
    d_max: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample n_samples first-passage times, with starting state drawn
    from the distribution ``pi0`` (one draw per sample).

    This is the distributional analogue of ``first_passage_times``: it
    matches the held-out empirical setting, where different test traces
    start from different initial latent clusters.
    """
    if rng is None:
        rng = np.random.default_rng()
    m = Q.shape[0]
    # cumulative transitions from each state (transient+absorbing).
    full = np.concatenate([Q, R_succ[:, None], R_fail[:, None]], axis=1)
    cum = np.cumsum(full, axis=1)
    pi0 = np.asarray(pi0, dtype=float).reshape(-1)
    pi0 = pi0 / pi0.sum()
    out = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        # Draw the starting state from pi0.
        s = int(rng.choice(m, p=pi0))
        for t in range(1, d_max + 1):
            u = rng.random()
            nxt = int(np.searchsorted(cum[s], u))
            if nxt == m or nxt == m + 1:   # absorbed
                out[i] = t
                break
            s = nxt
        else:
            out[i] = d_max
    return out

# We DO NOT reuse SS8's trace generator here.  That generator samples
# trace length from a bounded uniform before rolling out per-step
# features, so the resulting process is NOT an absorbing Markov chain and
# no fitted chain can match its first-passage CDF -- which would make a
# held-out KS test systematically fail regardless of pipeline quality.
# Instead, SS9 generates traces from a genuinely absorbing Markov chain
# (see ``make_markov_mast_corpus`` below).  This tests the pipeline's
# ability to recover the chain from noisy step-level observations.
from SS8_uncertainty_quantification import FRAMEWORKS


DATA_DIR = ROOT / "data" / "mast_derived"
FIG_DIR = ROOT / "figs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Seed chosen independently of SS7/SS8 to avoid any leakage.
SEED = 20260422_17
N_TRACES = 400        # per framework; 50/50 split -> 200 fit + 200 test
D_MAX = 50
N_MODEL_FPT = 8_000   # for KS-test reference distribution


# --------------------------- Stage 1: split corpus ---------------------------


def split_traces(
    traces: list, frac: float, rng: np.random.Generator
) -> tuple[list, list]:
    """Random 50/50 split of a trace corpus."""
    idx = np.arange(len(traces))
    rng.shuffle(idx)
    cut = int(len(traces) * frac)
    fit_idx = set(idx[:cut].tolist())
    fit_t = [traces[i] for i in range(len(traces)) if i in fit_idx]
    test_t = [traces[i] for i in range(len(traces)) if i not in fit_idx]
    return fit_t, test_t


# --------------------------- Stage 2: empirical stats ------------------------


def empirical_trace_length_and_outcome(
    traces: list[list[TraceStep]],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return (lengths, succeeded, n_abs, n_total) for a trace corpus.

    lengths[i]  = number of transient-state visits in trace i before absorption
                  (or L_i if the trace is censored -- these are masked out
                  of the CDF via ``succeeded``).
    succeeded[i] in {0, 1, nan} where nan -> censored.
    """
    lengths = []
    succ = []
    n_abs = 0
    for tr in traces:
        # Exclude the terminal step when counting transient visits.
        if tr and tr[-1].is_terminal:
            L = len(tr) - 1
            term = tr[-1].terminal_label
        else:
            L = len(tr)
            term = None
        lengths.append(L)
        if term == "success":
            succ.append(1.0)
            n_abs += 1
        elif term == "failure":
            succ.append(0.0)
            n_abs += 1
        else:
            succ.append(np.nan)
    return (
        np.asarray(lengths, dtype=int),
        np.asarray(succ, dtype=float),
        n_abs,
        len(traces),
    )


def empirical_rdc(lengths: np.ndarray, succ: np.ndarray, d_max: int) -> np.ndarray:
    """Empirical reliability-decay curve on test traces.

        \hat{\mathcal R}_{\mathrm{emp}}(d)
            = (# test traces absorbed as success with length <= d) / N_test

    Censored traces (succ == nan) are counted in the denominator; they
    cannot exceed numerator since their outcome is unknown.  This yields
    a CONSERVATIVE empirical RDC, which is the standard convention.
    """
    N = len(lengths)
    out = np.zeros(d_max + 1)
    for d in range(1, d_max + 1):
        mask = (succ == 1.0) & (lengths <= d)
        out[d] = mask.sum() / N
    return out


def empirical_first_passage_lengths(
    lengths: np.ndarray, succ: np.ndarray
) -> np.ndarray:
    """First-passage times from test traces.  Censored traces excluded."""
    mask = ~np.isnan(succ)
    return lengths[mask].astype(int)


# --------------------------- Stage 3: KS on FPT ------------------------------


def two_sample_ks(
    a: np.ndarray, b: np.ndarray
) -> tuple[float, float]:
    """Two-sample KS statistic + asymptotic p-value.

    Reimplemented locally to avoid a scipy dependency (cf. mcr.trace_to_chain).
    """
    a = np.sort(a.astype(float))
    b = np.sort(b.astype(float))
    xs = np.unique(np.concatenate([a, b]))
    Fa = np.searchsorted(a, xs, side="right") / a.size
    Fb = np.searchsorted(b, xs, side="right") / b.size
    D = float(np.max(np.abs(Fa - Fb)))
    n1, n2 = a.size, b.size
    en = np.sqrt(n1 * n2 / (n1 + n2))
    arg = (en + 0.12 + 0.11 / en) * D
    p = 2.0 * sum((-1) ** (j - 1) * np.exp(-2 * (j * arg) ** 2) for j in range(1, 101))
    p = max(0.0, min(1.0, p))
    return D, float(p)


# --------------------------- Stage 4: run one framework ----------------------


def run_framework(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    # Generate full corpus and split 50/50.
    corpus = make_markov_mast_corpus(name, seed=seed, n_traces=N_TRACES)
    fit_t, test_t = split_traces(corpus, frac=0.5, rng=rng)

    # Fit on fit_t only.
    chain = fit(fit_t, k_min=3, k_max=6, alpha=1.0)

    # Analytic RDC from fitted chain.
    # Use an initial *distribution* pi0 = empirical distribution over first
    # cluster labels on the fit set (i.e. not the single modal s0).
    first_labels = []
    pos = 0
    for tr in fit_t:
        # First non-terminal step of this trace is labels_flat[pos].
        first_labels.append(int(chain.labels[pos]))
        # Advance pos past this trace's transient steps.
        for s in tr:
            if not s.is_terminal:
                pos += 1
    first_labels = np.asarray(first_labels, dtype=int)
    pi0 = np.bincount(first_labels, minlength=chain.m).astype(float)
    pi0 = pi0 / pi0.sum()

    R_analytic = reliability_curve(
        chain.Q, chain.R_succ, s0=pi0, d_max=D_MAX
    )

    # Empirical RDC on held-out test traces.
    lengths_tr, succ_tr, n_abs_tr, n_tr = empirical_trace_length_and_outcome(test_t)
    R_empirical = empirical_rdc(lengths_tr, succ_tr, d_max=D_MAX)

    # L_inf RDC error.
    L_inf_rdc = float(np.max(np.abs(R_analytic - R_empirical)))

    # KS on FPT.
    emp_fpt = empirical_first_passage_lengths(lengths_tr, succ_tr)
    model_fpt = first_passage_times_pi0(
        chain.Q, chain.R_succ, chain.R_fail,
        pi0=pi0,
        n_samples=N_MODEL_FPT,
        rng=rng,
    )
    D_ks, p_ks = two_sample_ks(emp_fpt, model_fpt.astype(float))

    return {
        "framework": name,
        "n_fit": len(fit_t),
        "n_test": len(test_t),
        "n_test_absorbed": int(n_abs_tr),
        "m": int(chain.m),
        "first_order_preferred": bool(chain.first_order_preferred),
        "D_KS": D_ks,
        "p_KS": p_ks,
        "L_inf_RDC": L_inf_rdc,
        "R_analytic": R_analytic.tolist(),
        "R_empirical": R_empirical.tolist(),
        "d_grid": list(range(D_MAX + 1)),
    }


# --------------------------- Stage 5: outputs --------------------------------


def write_summary_table(results: list[dict], path: Path) -> None:
    lines = [
        "% Auto-generated by SS9_heldout_empirical.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{SS9: held-out recovery on seven controlled MAST-style corpora.}",
        "\\label{tab:heldout-ss9}",
        "\\small",
        "\\begin{tabular}{l c c c c c c}",
        "\\toprule",
        "\\textbf{Framework} & \\textbf{$n_{\\mathrm{fit}}$} "
        "& \\textbf{$n_{\\mathrm{test}}$} & \\textbf{$m$} "
        "& \\textbf{$D_{\\mathrm{KS}}$} & \\textbf{$p_{\\mathrm{KS}}$} "
        "& \\textbf{$L_\\infty^{\\mathrm{RDC}}$} \\\\",
        "\\midrule",
    ]
    cite = {"react": "yao2023react", "reflexion": "shinn2023reflexion",
            "cot_agent": "wang2024agentsurvey", "toolformer": "schick2023toolformer",
            "babyagi": "wang2024agentsurvey", "autogpt": "wang2024agentsurvey",
            "agentbench": "agentbench2024"}
    for i, res in enumerate(results):
        if i % 2 == 1:
            lines.append("\\tstriped")
        lines.append(
            f"{res['framework'].replace('_', chr(92)+'_')}"
            f"~\\cite{{{cite[res['framework']]}}} & "
            f"{res['n_fit']} & {res['n_test']} & {res['m']} & "
            f"{res['D_KS']:.3f} & {res['p_KS']:.3f} & {res['L_inf_RDC']:.3f} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")


def make_overlay_figure(results: list[dict], path: Path) -> None:
    """Small-multiple overlay of R_analytic vs R_empirical per framework."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping overlay figure.")
        return

    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.25 * cols, 1.85 * rows),
                             sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)
    d = np.arange(D_MAX + 1)
    for ax, res in zip(axes, results):
        Ra = np.asarray(res["R_analytic"])
        Re = np.asarray(res["R_empirical"])
        ax.plot(d, Ra, "-", color="C0", lw=1.4,
                label=r"$\mathcal{R}(d)$ (analytic)")
        ax.plot(d, Re, "--", color="C3", lw=1.2,
                label=r"$\hat{\mathcal{R}}_{\mathrm{emp}}(d)$")
        ax.set_title(
            f"{res['framework']}\n"
            rf"$L_\infty$={res['L_inf_RDC']:.3f}, $p$={res['p_KS']:.2f}",
            fontsize=8,
        )
        ax.set_xlim(0, D_MAX)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    # Hide unused axes.
    for ax in axes[len(results):]:
        ax.axis("off")
    # Single legend on the last used axis.
    axes[0].legend(fontsize=7, loc="lower right")
    for ax in axes[-cols:]:
        ax.set_xlabel("budget $d$", fontsize=8)
    axes[0].set_ylabel("reliability", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------- main ---------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run with fewer traces (smoke test).",
    )
    args = parser.parse_args()
    global N_TRACES
    if args.quick:
        N_TRACES = 120

    print("SS9 -- Held-out empirical validation")
    print(f"  N_TRACES = {N_TRACES} per framework, 50/50 fit/test split")
    print(f"  D_MAX    = {D_MAX}")
    results = []
    for i, name in enumerate(FRAMEWORKS):
        res = run_framework(name, seed=SEED + i)
        print(
            f"  {name:12s}  m={res['m']}  "
            f"D_KS={res['D_KS']:.3f}  p_KS={res['p_KS']:.3f}  "
            f"L_inf_RDC={res['L_inf_RDC']:.3f}  "
            f"n_test={res['n_test']} (abs={res['n_test_absorbed']})"
        )
        results.append(res)

    # Aggregate claims for the abstract / §VI-C.
    all_p = np.array([r["p_KS"] for r in results])
    all_Linf = np.array([r["L_inf_RDC"] for r in results])
    n_pass = int(np.sum(all_p > 0.05))
    summary = {
        "config": {
            "N_TRACES_per_framework": N_TRACES,
            "D_MAX": D_MAX,
            "N_MODEL_FPT": N_MODEL_FPT,
            "seed": SEED,
        },
        "aggregate": {
            "n_frameworks": len(results),
            "n_pass_KS_at_0p05": n_pass,
            "max_L_inf_RDC": float(all_Linf.max()),
            "median_L_inf_RDC": float(np.median(all_Linf)),
            "min_p_KS": float(all_p.min()),
            "median_p_KS": float(np.median(all_p)),
        },
        "frameworks": results,
    }
    print(
        f"\nAggregate: {n_pass}/{len(results)} pass KS at 0.05; "
        f"max L_inf RDC = {all_Linf.max():.3f}; "
        f"min p_KS = {all_p.min():.3f}"
    )

    (DATA_DIR / "SS9_heldout_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_summary_table(results, DATA_DIR / "SS9_heldout_summary.tex")
    make_overlay_figure(results, FIG_DIR / "fig_rdc_overlay.pdf")
    print(f"Wrote {DATA_DIR / 'SS9_heldout_summary.json'}")
    print(f"Wrote {DATA_DIR / 'SS9_heldout_summary.tex'}")
    print(f"Wrote {FIG_DIR / 'fig_rdc_overlay.pdf'}")
    print("[DONE] SS9 complete.")


if __name__ == "__main__":
    main()
