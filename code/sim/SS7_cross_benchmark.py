"""SS7_cross_benchmark.py — Broadened empirical validation using archetypes.

Validates the MCR-Agent framework across abstract models of three standard
LLM agent benchmarks: SWE-bench, tau-bench, and AgentBench.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Adjust path to find the `mcr` module
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from mcr.reliability import reliability_curve, asymptotic_reliability

def run_ss7():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_archetypes.json"
    
    with open(data_path, 'r') as f:
        archetypes = json.load(f)
        
    plt.figure(figsize=(7, 5))
    
    for bench_name, data in archetypes.items():
        Q = np.array(data["Q"])
        R_succ = np.array(data["R_succ"])
        R_fail = np.array(data["R_fail"])
        s0 = data["s0"]
        m = len(data["states"])
        
        # Check transience
        w, _ = np.linalg.eig(Q)
        rho = max(abs(w))
        
        print(f"--- {bench_name} ---")
        print(f"Num States : {m}")
        print(f"Spectral ρ : {rho:.4f}")
        
        r_inf = asymptotic_reliability(Q, R_succ, s0=s0)
        print(f"R_infty    : {r_inf:.4f}")
        
        rdc = reliability_curve(Q, R_succ, d_max=30, s0=s0)
        
        # rdc length is 31 (from 0 to 30)
        plt.plot(range(31), rdc, lw=2, marker='o', markersize=4, label=f"{bench_name} ($R_\\infty={r_inf:.3f}$)")

    plt.xlabel("Execution Steps $d$")
    plt.ylabel("Trajectory Reliability $R(d)$ (RDC)")
    plt.title("SS7: Cross-Benchmark Generality (Archetype RDC)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = Path(__file__).resolve().parent.parent.parent / "figs"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "fig_cross_benchmark.pdf"
    
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved RDC plot to {out_path}")

if __name__ == "__main__":
    run_ss7()
