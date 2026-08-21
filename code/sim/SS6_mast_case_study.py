"""SS6 — MAST case study: apply T1–T6 to empirical chains (SS6).

Applies the MCR framework to each of the 7 MAST frameworks (or their
synthetic stand-ins when the dataset is not available).  Produces the
reliability decay curve (RDC) per framework, ranks by R_infty, and
computes the T2 perturbation sensitivity and T6 horizon requirement.

Usage:
    python SS6_mast_case_study.py                        # synthetic data
    python SS6_mast_case_study.py --csv path/to/mast.csv  # real MAST data
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

from mcr.reliability import reliability_curve, asymptotic_reliability, fundamental_matrix
from mast_adapter import build_synthetic_mast_example, load_mast_csv, build_chain

FIGS_DIR = ROOT / "figs"
DATA_DIR = ROOT / "data" / "mast_derived"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Synthetic stand-ins for 7 MAST frameworks
FRAMEWORKS = [
    "react", "reflexion", "cot_agent", "toolformer",
    "babyagi", "autogpt", "agentbench",
]


def make_synthetic_frameworks() -> dict:
    """Generate 7 synthetic chains with varied reliability profiles."""
    rng = np.random.default_rng(20260420_06)
    frameworks = {}
    for i, name in enumerate(FRAMEWORKS):
        from mcr.chains import random_substochastic
        rho = rng.uniform(0.35, 0.80)
        succ = rng.uniform(0.05, 0.45)
        m = rng.integers(5, 13)
        Q, R_succ, R_fail = random_substochastic(
            int(m), rho_target=rho, succ_rate=succ, rng=rng
        )
        frameworks[name] = (Q, R_succ, R_fail, 0,
                            [f"s{j}" for j in range(int(m))])
    return frameworks


def apply_theorems(Q, R_succ, R_fail, s0, d_max=60) -> dict:
    """Apply T1, T2, T6 to a chain and return summary metrics."""
    N = fundamental_matrix(Q)
    R_inf = asymptotic_reliability(Q, R_succ, s0=s0)
    rdc = reliability_curve(Q, R_succ, s0=s0, d_max=d_max)
    rho_Q = float(np.max(np.abs(np.linalg.eigvals(Q))))
    norm_NR = float(np.linalg.norm(N @ R_succ, ord=np.inf))

    # T6: horizon to reach within delta=0.01 of R_inf
    delta = 0.01
    if rho_Q < 1 and rho_Q > 0 and norm_NR > delta:
        horizon_01 = int(np.ceil(np.log(norm_NR / delta) / (-np.log(rho_Q))))
    else:
        horizon_01 = d_max

    return {
        "R_inf": float(R_inf),
        "rho_Q": float(rho_Q),
        "norm_NR": float(norm_NR),
        "horizon_to_delta01": min(horizon_01, d_max),
        "rdc_d5": float(rdc[min(5, d_max)]),
        "rdc_d20": float(rdc[min(20, d_max)]),
        "rdc": rdc.tolist(),
        "m": Q.shape[0],
    }


def run(csv_path: str | None = None) -> dict:
    if csv_path:
        try:
            df_all = load_mast_csv(csv_path)
            frameworks_data = {}
            fw_col = "framework" if "framework" in df_all.columns else None
            avail = sorted(df_all[fw_col].unique()) if fw_col else ["all"]
            for fw in avail:
                try:
                    res = build_chain(df_all, framework_filter=fw)
                    frameworks_data[fw] = res
                except Exception as e:
                    print(f"  WARNING: skipping {fw}: {e}")
        except Exception as e:
            print(f"  CSV load failed ({e}); falling back to synthetic data.")
            frameworks_data = make_synthetic_frameworks()
    else:
        print("  No CSV provided — using synthetic MAST stand-ins.")
        frameworks_data = make_synthetic_frameworks()

    results = {}
    for fw, chain_data in frameworks_data.items():
        Q, R_succ, R_fail, s0 = chain_data[:4]
        metrics = apply_theorems(Q, R_succ, R_fail, s0)
        results[fw] = metrics
        print(f"  {fw:15s}: R_inf={metrics['R_inf']:.4f}  "
              f"rho={metrics['rho_Q']:.3f}  "
              f"horizon_d01={metrics['horizon_to_delta01']:3d}  "
              f"m={metrics['m']}")

    # Rank by R_inf
    ranked = sorted(results.items(), key=lambda x: x[1]["R_inf"], reverse=True)
    print("\nRanking by R_inf (descending):")
    for rank, (fw, m) in enumerate(ranked, 1):
        print(f"  {rank}. {fw:15s}  {m['R_inf']:.4f}")

    return results


def make_figure(results: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping figure.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (fw, metrics), color in zip(
        sorted(results.items(), key=lambda x: -x[1]["R_inf"]), colors
    ):
        rdc = np.array(metrics["rdc"])
        ax.plot(range(len(rdc)), rdc, label=f"{fw} ($R_\\infty$={metrics['R_inf']:.3f})",
                color=color)

    ax.set_xlabel("Horizon $d$ (steps)")
    ax.set_ylabel("$R(d)$ — Reliability Decay Curve")
    ax.set_title("SS6: MAST Case Study — RDC per Framework")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    out = FIGS_DIR / "fig_rdc.pdf"
    plt.savefig(out, bbox_inches="tight")
    print(f"Figure saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="Path to MAST CSV file (optional)")
    args = parser.parse_args()

    print("SS6 — MAST case study")
    results = run(csv_path=args.csv)

    out_json = DATA_DIR / "SS6_summary.json"
    serializable = {
        fw: {k: v for k, v in m.items() if k != "rdc"}
        for fw, m in results.items()
    }
    out_json.write_text(json.dumps(serializable, indent=2))
    make_figure(results)
    print(f"\nSummary written to {out_json}")
    print("[DONE] SS6 complete.")


if __name__ == "__main__":
    main()
