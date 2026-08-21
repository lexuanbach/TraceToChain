# `mcr`: TraceToChain — Auditing LLM-Agent Reliability with Absorbing Markov Chains

Reproducibility artifact for the ISSRE 2026 paper
*TraceToChain: Auditing LLM-Agent Reliability with Absorbing Markov Chains*
(`paper/main.pdf`).

The `mcr` package and the scripts under `code/sim/` take a corpus of agent
execution traces, fit an absorbing discrete-time Markov chain
`(Q, R_succ, R_fail)` via the TraceToChain pipeline (Alg. 1), gate every
downstream quantity behind the composite AIC ∧ KS certificate (Alg. 2), and
report the closed-form reliability `R(d) = e_s0^T (I − Q^d) N R_succ`, the
asymptotic reliability `R_inf = e_s0^T N R_succ` with `N = (I − Q)^{-1}`, and
the derived metrics (pass@k, pass^k, RDC, MTTA) with Dirichlet-posterior and
bootstrap intervals.

## Requirements

- Python ≥ 3.10 with `numpy ≥ 1.22`, `scipy`, `matplotlib`, `pandas`
- `pdflatex` + `bibtex` (only to rebuild the paper)
- PRISM 4.7+ on `$PATH` (optional; SS5 is skipped otherwise)

No GPU, no network access, no proprietary data. Every script seeds its RNG
explicitly (`data/synthetic/seeds.json`; `SEED` in each real-trace script).

## Quick start

```bash
pip install -e ./code

bash reproduce.sh --quick    # ~2 min   smoke test (see caveat below)
bash reproduce.sh            # ~25 min  synthetic + MAST studies (SS1–SS9)
bash reproduce.sh --full     # ~2.5 h   everything, incl. real traces (SS10–SS15)
```

`--quick` runs reduced sample sizes, so its numbers deliberately do **not**
match the paper; it snapshots `data/` and `figs/` and restores them on exit, so
it can never leave the artifact disagreeing with the manuscript. The standard
and `--full` tiers regenerate `data/` and `figs/` in place — use
`git diff --stat data/` to compare a fresh run against the shipped results.

## Studies and where their numbers appear

| Study | Script | What it establishes | Paper |
|---|---|---|---|
| SS1  | `SS1_cf_vs_mc.py` | closed form vs. Monte Carlo, all sizes under the 5% gate | §VII-A |
| SS2  | `SS2_perturbation.py` | Neumann perturbation bound vs. ground truth, ε ∈ [0, 0.28] | §VII-A (artifact) |
| SS3  | `SS3_correlation.py` | Jensen gap `pass^k_mix ≥ R_inf^k` grows with k and Var p(ξ) | §VII-A (artifact) |
| SS4  | `SS4_nhpp_limit.py` | Goel–Okumoto limit, KS p > 0.05 on all six scaling points | §VII-A (artifact) |
| SS5  | `SS5_prism_cross.py` | PRISM agreement to 1e-8 (optional) | artifact only |
| SS6  | `SS6_mast_case_study.py` | MAST reliability ranking, `R_inf ∈ [0.058, 0.450]` | §VIII-A |
| SS7  | `SS7_goodness_of_fit.py` | Type-I/II behaviour of the composite certificate | §VII-B |
| SS7b | `SS7_cross_benchmark.py` | cross-benchmark archetypes | artifact only |
| SS8  | `SS8_uncertainty_quantification.py` | Dirichlet + bootstrap intervals | Tables II–III |
| SS9  | `SS9_heldout_empirical.py` | strict 50/50 held-out recovery, 7/7 pass | Table IV, Fig. 2 |
| SS10 | `SS10_swebench_real.py` | real SWE-agent fit; certificate **rejects** | Table V |
| SS11 | `SS11_ablation.py` | 18-configuration clustering ablation | Table VI |
| SS12 | `SS12_censoring.py` | censoring 0–50% on a known chain | §XI |
| SS13 | `SS13_bootstrap_paths.py` | fast vs. full re-cluster bootstrap | §XI |
| SS14 | `SS14_higher_order.py` | order-k lifts; `R_inf` accepted, timing rejected | Table VII, Fig. 3 |
| SS15 | `SS15_taubench.py` | τ-bench replication across 2 models × 2 domains | Fig. 4 |

### Headline numbers (from the shipped JSONs)

**SS7 — composite certificate.** Markov ground truth: null retained in 30/30
corpora at α = 0.05 (conservative rather than exactly nominal). Second-order
ground truth: KS layer alone rejects 0%, AIC layer rejects 100% with
ΔAIC ∈ [+979, +1614], composite rejects 100%. MAST self-consistency: all 7
accepted, KS p ∈ {0.320, 0.374, 0.541, 0.828, 0.910, 0.994, 1.000}.

> ΔAIC is `AIC(1st) − AIC(2nd)`; the first-order model is preferred iff ΔAIC < 0.

**SS9 — held-out validation** (`data/mast_derived/SS9_heldout_summary.json`):

| Framework | m | D_KS | p_KS | L∞^RDC |
|---|---:|---:|---:|---:|
| react | 5 | 0.020 | 1.000 | 0.0476 |
| reflexion | 5 | 0.033 | 0.984 | 0.0525 |
| cot_agent | 5 | 0.023 | 1.000 | 0.0184 |
| toolformer | 5 | 0.030 | 0.994 | 0.0523 |
| babyagi | 6 | 0.045 | 0.827 | 0.0535 |
| autogpt | 6 | 0.036 | 0.960 | 0.0387 |
| agentbench | 5 | 0.030 | 0.995 | 0.0212 |

7/7 pass at α = 0.05; min p_KS = 0.827, max L∞ = 0.0535, median 0.0476.

**SS10/SS14 — real SWE-agent traces.** 479 SWE-bench Verified instances
(324 resolved / 155 not; 29,504 steps). The clustered fit recovers m = 7
(silhouette 0.801) but the certificate **rejects**: ΔAIC = +817.2, held-out
KS p < 1e-4, L∞^RDC = 0.260. SS14 on the full corpus: first-order
`R_inf = 0.6565` vs. held-out pass rate `0.6833` (within 0.027); the timing is
rejected at every order k ∈ {1, 2, 3}.

**SS11 — ablation.** All 18 configurations REJECT; m ranges 6–10;
`R_inf ∈ [0.660, 0.674]`; KS p < 1e-3 everywhere. Three configurations
(category-only featurization at k_max = 10) have ΔAIC < 0, i.e. the AIC layer
alone would accept a first-order chain and only the held-out KS layer rejects.

**SS12/SS13.** Censoring 0–50% leaves relative bias in [−4.2%, −2.6%]
(uncensored baseline −3.3%). The full re-cluster bootstrap gives intervals
≈4.7× wider than the fixed-clustering path (0.158 vs. 0.034).

**SS15 — τ-bench.** 1,980 episodes over Claude-3.5-Sonnet and GPT-4o × {retail,
airline}. Mean |R_inf − pass rate| = 0.0075 across the four runs; pooled
0.595 vs. 0.597.

## Layout

```
code/
├── mcr/                                # installable package
│   ├── reliability.py                  # R(d), R_inf, N = (I − Q)^-1
│   ├── perturb.py                      # row-rank ΔQ construction
│   ├── simulate.py                     # Monte Carlo baseline
│   ├── chains.py                       # synthetic chain generators
│   ├── trace_to_chain.py               # Alg. 1 (fit) + Alg. 2 (goodness_of_fit)
│   └── uncertainty.py                  # Dirichlet posterior + bootstrap CIs
├── sim/                                # SS1–SS15 study scripts
│   ├── mast_adapter.py                 # MAST trace loader
│   ├── swebench_adapter.py             # .traj → TraceStep featurizer
│   ├── _build_corpus.py                # assembles data/swebench_real/corpus.json
│   └── harvest_taubench.js             # τ-bench trajectory extractor
└── pyproject.toml
data/
├── synthetic/                          # SS1–SS5, SS7 summaries + seeds.json
├── mast_derived/                       # SS6, SS8, SS9 summaries + LaTeX tables
├── swebench_real/                      # corpus.json + SS10–SS14 summaries
└── tau_bench/                          # τ-bench sufficient statistics + SS15
figs/                                   # every figure the paper includes
proofs/                                 # full proofs T1–T6 (sketched in the body)
paper/                                  # LaTeX source + main.pdf
reproduce.sh
```

## Data provenance

- **`data/swebench_real/corpus.json`** — derived from the publicly released
  `.traj` files of a SWE-agent submission to the SWE-bench Verified
  leaderboard (SWE-agent driving Claude-4-Sonnet). Only per-step action
  categories and the `resolved` verdict are retained; no trajectory text.
  `harvest_swebench.js` + `_build_corpus.py` rebuild it.
- **`data/tau_bench/`** — per-domain sufficient statistics (transition counts
  and success-length distributions) extracted from the publicly released
  τ-bench trajectory archive with `harvest_taubench.js`. The MLE and the
  first-passage KS test depend only on these, so raw trajectories are not
  redistributed.
- **`data/mast_derived/`** — transition matrices derived from public MAST
  summaries for seven agent frameworks.

## Paper ↔ code map

| Paper | Implementation |
|---|---|
| Prop. 1 (closed-form `R(d)`) | `mcr.reliability.reliability` |
| Prop. 2 (perturbation bound) | `mcr.perturb.perturb` |
| Thm. 3 (metric unification) | `mcr.reliability.asymptotic_reliability` |
| Thm. 4 / Cor. 5 (correlated trials) | `sim/SS3_correlation.py` |
| Props. 6–8 (shape, NHPP, approach rate) | `proofs/T4.tex`–`T6.tex`, `sim/SS4_nhpp_limit.py` |
| Alg. 1 (TraceToChain) | `mcr.trace_to_chain.fit` |
| Alg. 2 (composite GoF) | `mcr.trace_to_chain.goodness_of_fit` |
| §IV-D (uncertainty) | `mcr.uncertainty` |
| Fig. 2 (`fig_rdc_overlay`) | `sim/SS9_heldout_empirical.py` |
| Fig. 3 (`fig_ss14_realfit`) | `sim/SS14_higher_order.py` |
| Fig. 4 (calibration) | TikZ in `paper/sections/16_ss15_taubench.tex`, data from `sim/SS15_taubench.py` |

## Reproducibility notes

Each script writes a JSON summary next to its figure; those JSONs are the
ground truth for every numeric claim in the paper, and the LaTeX tables under
`data/*/` are generated directly from them (the paper `\input`s them, so the
manuscript cannot drift from the data).

The pipeline is deterministic given the seeds. Two caveats worth knowing:
Ward clustering on the real corpus is the dominant cost (SS11 fits 18 chains,
≈75 min), and the Layer-1 KS test draws its reference distribution by
simulation (N = 8,000 in SS9, 20,000 on the real corpora) under a fixed seed —
it is a genuine two-sample test, reproducible but not analytic. Floating-point
variation across BLAS versions (< 1e-10) does not affect any reported digit.
