"""SS3 — pass^k correlation effect (Theorems T3 and T3').

Constructs latent-xi chains where xi is binary with Pr[xi=1]=q.
Compares pass^k under the i.i.d. model (T3: R_inf^k) against the
true mixture (T3': E[p(xi)^k]) and reports the gap.

Usage:
    python SS3_correlation.py
    python SS3_correlation.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from mcr.reliability import asymptotic_reliability
from mcr.chains import random_substochastic

SEED = 20260420_03
K_VALS = [1, 2, 3, 5, 10, 20]
Q_VALS = [0.1, 0.3, 0.5, 0.7, 0.9]  # Pr[xi=1]

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "synthetic"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def latent_pass_metrics(p0: float, p1: float, q: float, k_vals):
    """Compute T3 (iid) and T3' (mixture) pass^k and pass@k.

    Args:
        p0: R_inf for xi=0 branch
        p1: R_inf for xi=1 branch
        q:  Pr[xi=1]
        k_vals: list of k to evaluate
    """
    R_inf_marginal = q * p1 + (1 - q) * p0  # E[p(xi)]

    results = []
    for k in k_vals:
        # T3 (i.i.d.)
        passk_iid = R_inf_marginal ** k
        passat_iid = 1 - (1 - R_inf_marginal) ** k

        # T3' (mixture)
        passk_mix = q * p1 ** k + (1 - q) * p0 ** k  # E[p(xi)^k]
        passat_mix = 1 - (q * (1 - p1) ** k + (1 - q) * (1 - p0) ** k)

        results.append({
            "k": k,
            "passk_iid": float(passk_iid),
            "passk_mix": float(passk_mix),
            "passk_gap": float(passk_mix - passk_iid),
            "passat_iid": float(passat_iid),
            "passat_mix": float(passat_mix),
            "passat_gap": float(passat_mix - passat_iid),
        })
    return results


def run() -> dict:
    rng = np.random.default_rng(SEED)
    m = 5
    # Generate two chains: xi=0 (low-success) and xi=1 (high-success)
    Q0, R0, Rf0 = random_substochastic(m, rho_target=0.6, succ_rate=0.05, rng=rng)
    Q1, R1, Rf1 = random_substochastic(m, rho_target=0.5, succ_rate=0.35, rng=rng)
    p0 = asymptotic_reliability(Q0, R0, s0=0)
    p1 = asymptotic_reliability(Q1, R1, s0=0)
    print(f"  p(xi=0)={p0:.4f}  p(xi=1)={p1:.4f}")

    all_results = {}
    for q in Q_VALS:
        R_inf_marginal = q * p1 + (1 - q) * p0
        print(f"  q=Pr[xi=1]={q:.1f}  R_inf_marginal={R_inf_marginal:.4f}")
        metrics = latent_pass_metrics(p0, p1, q, K_VALS)
        for row in metrics:
            print(f"    k={row['k']:2d}: passk_gap={row['passk_gap']:+.4f}  "
                  f"passat_gap={row['passat_gap']:+.4f}")
        all_results[str(q)] = metrics

    return {"p0": float(p0), "p1": float(p1), "q_vals": Q_VALS,
            "k_vals": K_VALS, "results": all_results}


def make_figure(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(Q_VALS)))

    for ax_idx, metric in enumerate(["passk_gap", "passat_gap"]):
        ax = axes[ax_idx]
        for color, q in zip(colors, Q_VALS):
            rows = data["results"][str(q)]
            gaps = [r[metric] for r in rows]
            ax.plot(K_VALS, gaps, "o-", color=color, label=f"q={q:.1f}")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("$k$")
        label = "pass$^k$ gap" if metric == "passk_gap" else "pass@$k$ gap"
        ax.set_ylabel(f"{label}: mixture $-$ i.i.d.")
        ax.set_title(f"SS3 Correlation effect — {label}")
        ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    out = FIGS_DIR / "fig_correlation_gap.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    print("SS3 — pass^k correlation effect")
    data = run()
    (DATA_DIR / "SS3_summary.json").write_text(json.dumps(data, indent=2))
    make_figure(data)
    print("[DONE] SS3 complete.")


if __name__ == "__main__":
    main()
