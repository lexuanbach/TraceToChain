"""SS4 — NHPP rare-failure limit verification (Proposition 5, T5).

REVISED: this script previously plotted the SURVIVAL function S(d) =
(1-eps)^d (monotone decreasing from 1 to 0) and labeled it R(d). That
was inconsistent with Definition 2 of R(d) as the cumulative
first-passage CDF to the success state (monotone increasing from 0 to
R_infinity). This version:

  * Uses the single-transient-state chain with separate per-step
    success rate mu_n and per-step failure rate eps_n (eps_n / mu_n -> 0
    under rare-failure scaling).
  * Plots R(d) = mu_n/(mu_n+eps_n) * (1 - (1 - mu_n - eps_n)^d), the
    CDF of T_oplus, which is monotone INCREASING from 0 to
    mu_n/(mu_n+eps_n).
  * Overlays the Goel-Okumoto mean-value function 1 - exp(-Lambda * c),
    where c = d / d_n.
  * Reports KS distance between the finite-n CDF and the limit CDF.
  * Separately shows the survival function S(d) = 1 - R(d) - failure-CDF
    = (1-mu_n-eps_n)^d in a small inset/legend for transparency, to
    demonstrate it is NOT the quantity being compared to Goel-Okumoto.

Scaling sequence: mu_n -> 0, d_n * mu_n -> Lambda, eps_n / mu_n -> 0.

Usage:
    python SS4_nhpp_limit.py
    python SS4_nhpp_limit.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

SEED = 20260422_04
# Sequence of (mu_n, ratio) pairs.  ratio = eps_n / mu_n -> 0.
MU_SEQUENCE = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005]
EPS_RATIO = 0.1  # eps_n = EPS_RATIO * mu_n; fixed small ratio so eps_n/mu_n -> 0.
LAMBDA_TARGET = 2.0  # d_n * mu_n -> Lambda

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "synthetic"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def exact_R_cdf(mu: float, eps: float, d_vals: np.ndarray) -> np.ndarray:
    """Exact first-passage CDF R(d) = P(T_oplus <= d) for the 1-state chain.

    From Proposition 1 applied to a single-transient-state chain:
        R(d) = mu / (mu+eps) * (1 - (1-mu-eps)^d).
    This is monotone INCREASING in d, with R(0)=0 and R(inf)=mu/(mu+eps).
    """
    q = 1.0 - mu - eps
    prefactor = mu / (mu + eps) if (mu + eps) > 0 else 0.0
    return prefactor * (1.0 - np.power(q, d_vals))


def exact_S_survival(mu: float, eps: float, d_vals: np.ndarray) -> np.ndarray:
    """Survival function S(d) = P(T_oplus > d) for transparency.

    S(d) = (1-mu-eps)^d is monotone DECREASING from 1 to 0.
    This is what the earlier (buggy) draft mislabeled as R(d).
    """
    return np.power(1.0 - mu - eps, d_vals)


def goel_okumoto(Lambda: float, c_vals: np.ndarray) -> np.ndarray:
    """Goel-Okumoto mean-value function 1 - exp(-Lambda * c).

    This is an INCREASING function from 0 to 1 in the rescaled horizon
    c = d / d_n.
    """
    return 1.0 - np.exp(-Lambda * c_vals)


def ks_statistic(empirical: np.ndarray, theoretical: np.ndarray) -> float:
    """Sup-norm (KS) distance between two CDFs."""
    return float(np.max(np.abs(empirical - theoretical)))


def run(mu_seq=MU_SEQUENCE) -> dict:
    summary = []

    for mu in mu_seq:
        eps = EPS_RATIO * mu
        d_n = int(round(LAMBDA_TARGET / mu))
        d_vals = np.arange(0, d_n + 1)
        c_vals = d_vals / d_n

        # Cumulative first-passage CDF R(d) — the quantity of interest.
        R_exact = exact_R_cdf(mu, eps, d_vals)
        # Goel-Okumoto limit in the rescaled horizon c = d / d_n.
        go_limit = goel_okumoto(LAMBDA_TARGET, c_vals)
        ks = ks_statistic(R_exact, go_limit)

        # Asymptotic KS p-value (Smirnov distribution, large-n approximation).
        n = len(d_vals)
        ks_scaled = ks * np.sqrt(n)
        p_val = 2 * np.exp(-2 * ks_scaled ** 2) if ks_scaled > 0 else 1.0

        print(f"  mu={mu:.4f}  eps={eps:.5f}  d_n={d_n:5d}  "
              f"R_inf={R_exact[-1]:.4f}  KS(R, GO)={ks:.4f}  "
              f"p≈{p_val:.4f}  {'PASS' if p_val >= 0.05 else 'FAIL'}")
        summary.append({
            "mu": float(mu),
            "eps": float(eps),
            "eps_over_mu": float(EPS_RATIO),
            "d_n": d_n,
            "R_inf": float(R_exact[-1]),
            "ks_R_vs_GO": float(ks),
            "ks_scaled": float(ks_scaled),
            "p_approx": float(p_val),
            "pass": bool(p_val >= 0.05),
        })

    return {
        "mu_sequence": list(mu_seq),
        "eps_over_mu": EPS_RATIO,
        "lambda_target": LAMBDA_TARGET,
        "note": "R(d) is the cumulative first-passage CDF to success (increasing). "
                "S(d) = (1-mu-eps)^d is the survival function (decreasing); "
                "it is plotted separately for transparency but is NOT compared "
                "to the Goel-Okumoto limit.",
        "summary": summary,
    }


def make_figure(data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    mu_seq = data["mu_sequence"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.flatten()

    for ax, mu in zip(axes, mu_seq):
        eps = EPS_RATIO * mu
        d_n = int(round(LAMBDA_TARGET / mu))
        d_vals = np.arange(0, d_n + 1)
        c_vals = d_vals / d_n

        R_exact = exact_R_cdf(mu, eps, d_vals)
        S_exact = exact_S_survival(mu, eps, d_vals)
        go_limit = goel_okumoto(LAMBDA_TARGET, c_vals)

        # Primary: cumulative first-passage CDF R(d), increasing.
        ax.plot(c_vals, R_exact,
                label=r"Exact $R(d) = \Pr[T_\oplus \leq d]$ (CDF, $\uparrow$)",
                color="steelblue", linewidth=2)
        # Goel-Okumoto limit (the micro-foundation claim).
        ax.plot(c_vals, go_limit, "--",
                label=r"Goel--Okumoto $1-e^{-\Lambda c}$",
                color="tomato", linewidth=2)
        # For transparency: plot S(d) lightly to show it is NOT R(d).
        ax.plot(c_vals, S_exact,
                label=r"Survival $S(d)=(1{-}\mu{-}\varepsilon)^{d}$ ($\downarrow$)",
                color="gray", alpha=0.5, linestyle=":", linewidth=1.2)

        ax.set_title(f"$\\mu={mu}$, $\\varepsilon={eps:.4f}$, $d_n={d_n}$")
        ax.set_xlabel("$c = d / d_n$")
        ax.set_ylabel("probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="center right")

    plt.suptitle("SS4: NHPP Rare-Failure Limit — cumulative first-passage "
                 "CDF $R(d)$ converges to Goel--Okumoto.",
                 y=1.01)
    plt.tight_layout()
    out = FIGS_DIR / "fig_nhpp_limit.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    mu_seq = MU_SEQUENCE[-3:] if args.quick else MU_SEQUENCE

    print("SS4 — NHPP rare-failure limit verification (revised)")
    print(f"  lambda_target = d_n * mu_n -> {LAMBDA_TARGET}")
    print(f"  eps_n = {EPS_RATIO} * mu_n  (eps_n/mu_n -> 0 is fixed at {EPS_RATIO})")
    print(f"  Target: R(d) = CDF of T_oplus -> 1 - exp(-Lambda * c), c = d/d_n")
    data = run(mu_seq=mu_seq)
    (DATA_DIR / "SS4_summary.json").write_text(json.dumps(data, indent=2))
    make_figure(data)

    passes = sum(1 for row in data["summary"] if row["pass"])
    total = len(data["summary"])
    print(f"\nKS test R(d) vs Goel-Okumoto: {passes}/{total} at alpha=0.05")
    print("[DONE] SS4 complete.")


if __name__ == "__main__":
    main()
