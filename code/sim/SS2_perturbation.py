"""SS2 — Perturbation bound tightness (Theorem T2).

For 100 random chains, sweeps eps in [0, 0.3] (21 values), compares the
analytic T2 upper bound against the true |R_inf(eps) - R_inf(0)|.
Reports the ratio (bound / true_gap) and produces fig_perturbation.pdf.

Usage:
    python SS2_perturbation.py
    python SS2_perturbation.py --quick   # 10 chains only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from mcr.reliability import asymptotic_reliability, fundamental_matrix
from mcr.chains import random_substochastic
from mcr.perturb import perturb

SEED = 20260420_02
EPS_VALS = np.linspace(0, 0.28, 21)  # stay below rho(Q) < 1
N_CHAINS = 100

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "synthetic"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def make_T2_bound(Q0, R_succ, Delta, eps_vals):
    """Compute the T2 analytic upper bound for each eps."""
    N0 = fundamental_matrix(Q0)
    norm_N0 = np.linalg.norm(N0, ord=np.inf)
    norm_Delta = np.linalg.norm(Delta, ord=np.inf)
    norm_Rsucc = float(np.max(np.abs(R_succ)))
    bounds = []
    for eps in eps_vals:
        denom = 1 - eps * norm_N0 * norm_Delta
        if denom <= 0:
            bounds.append(np.inf)
        else:
            C = norm_N0 ** 3 * norm_Delta ** 2 * norm_Rsucc / denom
            bound = eps * norm_N0 ** 2 * norm_Delta * norm_Rsucc + C * eps ** 2
            bounds.append(float(bound))
    return np.array(bounds)


def run(n_chains: int = N_CHAINS) -> dict:
    rng_master = np.random.default_rng(SEED)
    all_ratios = []  # ratio = bound / true_gap (at eps=0.1)
    chain_results = []

    for chain_idx in range(n_chains):
        seed = int(rng_master.integers(0, 2**32))
        rng = np.random.default_rng(seed)
        m = 10
        Q0, R_succ, R_fail = random_substochastic(
            m, rho_target=rng.uniform(0.4, 0.85),
            density=rng.uniform(0.4, 1.0),
            succ_rate=rng.uniform(0.05, 0.3),
            rng=rng,
        )
        # Build a valid Delta: remove mass proportionally from Q rows, route to fail
        raw_delta = -rng.random((m, m)) * Q0  # negative: removes mass
        # Normalize so that each row of Delta sums to some negative value
        row_remove = rng.uniform(0.01, 0.1, size=m)  # fraction of row mass removed
        for i in range(m):
            row_sum_Q = Q0[i].sum()
            if row_sum_Q < 1e-10:
                raw_delta[i] *= 0
            else:
                raw_delta[i] = -Q0[i] * row_remove[i] / row_sum_Q
        Delta = raw_delta  # each row sums to -row_remove[i]

        true_gaps = []
        for eps in EPS_VALS:
            try:
                Qe, Re_succ, Re_fail = perturb(Q0, R_succ, R_fail, eps, Delta,
                                                reroute_to="fail")
                r0 = asymptotic_reliability(Q0, R_succ, s0=0)
                re = asymptotic_reliability(Qe, Re_succ, s0=0)
                true_gaps.append(abs(re - r0))
            except Exception:
                true_gaps.append(np.nan)

        bounds = make_T2_bound(Q0, R_succ, Delta, EPS_VALS)
        true_gaps = np.array(true_gaps)

        # Ratio at eps=0.1 (index 7 in 21-step linspace from 0 to 0.28)
        idx = np.argmin(np.abs(EPS_VALS - 0.1))
        gap_at_01 = true_gaps[idx]
        bnd_at_01 = bounds[idx]
        if gap_at_01 > 1e-12:
            ratio = bnd_at_01 / gap_at_01
            all_ratios.append(ratio)

        chain_results.append({
            "chain_idx": chain_idx,
            "true_gaps": true_gaps.tolist(),
            "bounds": bounds.tolist(),
        })

    ratios = np.array(all_ratios)
    print(f"Bound / true gap at eps=0.1:  "
          f"median={np.median(ratios):.2f}  "
          f"p10={np.percentile(ratios, 10):.2f}  "
          f"p90={np.percentile(ratios, 90):.2f}")
    return {"ratios": ratios.tolist(), "eps_vals": EPS_VALS.tolist(),
            "chain_results": chain_results}


def make_figure(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    eps = np.array(data["eps_vals"])
    chains = data["chain_results"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    sample = chains[:10]  # show 10 example chains
    for c in sample:
        gaps = np.array(c["true_gaps"])
        bnds = np.array(c["bounds"])
        ax.plot(eps, gaps, color="steelblue", alpha=0.6, linewidth=1)
        ax.plot(eps, bnds, color="tomato", alpha=0.4, linewidth=1, linestyle="--")
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("$|R_\\infty(\\varepsilon) - R_\\infty(0)|$")
    ax.set_title("True gap (blue) vs. T2 bound (red dashed)")

    ax = axes[1]
    ratios = np.array(data["ratios"])
    ax.hist(ratios[np.isfinite(ratios) & (ratios < 100)], bins=30, color="steelblue",
            edgecolor="white")
    ax.axvline(1.0, color="red", linestyle="--", label="Ratio=1 (tight)")
    ax.set_xlabel("Bound / true gap at $\\varepsilon=0.1$")
    ax.set_ylabel("Count")
    ax.set_title("Tightness of T2 bound (100 chains)")
    ax.legend()

    plt.tight_layout()
    out = FIGS_DIR / "fig_perturbation.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    n = 10 if args.quick else N_CHAINS

    print("SS2 — Perturbation bound tightness")
    data = run(n_chains=n)
    (DATA_DIR / "SS2_summary.json").write_text(
        json.dumps({"ratios": data["ratios"], "eps_vals": data["eps_vals"]}, indent=2)
    )
    make_figure(data)
    print("[DONE] SS2 complete.")


if __name__ == "__main__":
    main()
