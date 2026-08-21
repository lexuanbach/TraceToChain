"""SS1 — Closed-form reliability vs. Monte Carlo validation.

500 random substochastic chains across sizes m in {5, 10, 50, 100, 500};
10^5 Monte Carlo trajectories per chain.  Reports max absolute error,
compares to 3*sigma sampling noise, and produces fig_ss1_cf_vs_mc.pdf.

Gate G2 criterion: max error < 5% (0.05) for all sizes.

Usage:
    python SS1_cf_vs_mc.py                 # runs and saves figure
    python SS1_cf_vs_mc.py --quick         # 50 chains, 10^4 trajectories
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Resolve paths regardless of CWD
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from mcr.reliability import asymptotic_reliability
from mcr.simulate import monte_carlo_reliability
from mcr.chains import random_substochastic

SEED_MASTER = 20260420_01
SIZES = [5, 10, 50, 100, 500]
N_CHAINS = 500   # per size group
N_TRAJ = 100_000

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "synthetic"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def run(n_chains: int = N_CHAINS, n_traj: int = N_TRAJ) -> dict:
    rng_master = np.random.default_rng(SEED_MASTER)
    results: dict[int, list[float]] = {m: [] for m in SIZES}

    t0 = time.perf_counter()
    for m in SIZES:
        print(f"  m={m:4d}: ", end="", flush=True)
        for _ in range(n_chains):
            seed = int(rng_master.integers(0, 2**32))
            rng = np.random.default_rng(seed)
            Q, R_succ, R_fail = random_substochastic(
                m, rho_target=rng.uniform(0.4, 0.9),
                density=rng.uniform(0.3, 1.0),
                succ_rate=rng.uniform(0.05, 0.4),
                rng=rng,
            )
            r_cf = asymptotic_reliability(Q, R_succ, s0=0)
            mc = monte_carlo_reliability(Q, R_succ, R_fail, s0=0, n=n_traj, rng=rng)
            err = abs(r_cf - mc["mean"])
            results[m].append(err)
        arr = np.array(results[m])
        print(
            f"max_err={arr.max():.4f}  mean_err={arr.mean():.4f}  "
            f"p99={np.percentile(arr, 99):.4f}"
        )

    elapsed = time.perf_counter() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")
    return results


def save_results(results: dict) -> None:
    summary = {
        str(m): {
            "max_err": float(np.max(v)),
            "mean_err": float(np.mean(v)),
            "p99_err": float(np.percentile(v, 99)),
            "n_chains": len(v),
        }
        for m, v in results.items()
    }
    out_path = DATA_DIR / "SS1_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {out_path}")


def make_figure(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure generation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: box plot of absolute errors per size
    ax = axes[0]
    data_for_box = [results[m] for m in SIZES]
    ax.boxplot(data_for_box, labels=[str(m) for m in SIZES], showfliers=False)
    ax.axhline(0.05, color="red", linestyle="--", label="5% gate")
    ax.set_xlabel("Chain size $m$")
    ax.set_ylabel("$|R_\\infty^{\\rm cf} - \\hat{R}_\\infty^{\\rm MC}|$")
    ax.set_title("SS1: Closed-form vs. Monte Carlo (all sizes)")
    ax.legend()

    # Right: max error vs m
    ax = axes[1]
    max_errs = [np.max(results[m]) for m in SIZES]
    ax.plot(SIZES, max_errs, "o-", color="steelblue")
    ax.axhline(0.05, color="red", linestyle="--", label="5% gate")
    ax.set_xscale("log")
    ax.set_xlabel("Chain size $m$ (log scale)")
    ax.set_ylabel("Max absolute error")
    ax.set_title("Max error across 500 chains")
    ax.legend()

    plt.tight_layout()
    out_path = FIGS_DIR / "fig_ss1_cf_vs_mc.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SS1: CF vs MC validation")
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: 50 chains, 10^4 trajectories")
    args = parser.parse_args()

    n_chains = 50 if args.quick else N_CHAINS
    n_traj = 10_000 if args.quick else N_TRAJ

    print(f"SS1 — Closed-form vs. Monte Carlo")
    print(f"  chains per size: {n_chains}, trajectories per chain: {n_traj:,}")
    results = run(n_chains=n_chains, n_traj=n_traj)
    save_results(results)
    make_figure(results)

    # Gate G2 check
    failed = [m for m, v in results.items() if np.max(v) >= 0.05]
    if failed:
        print(f"\n[FAIL] Gate G2 violated for sizes: {failed}")
        sys.exit(1)
    else:
        print("\n[PASS] Gate G2: all max errors < 5%")


if __name__ == "__main__":
    main()
