#!/usr/bin/env bash
# reproduce.sh -- end-to-end reproduction for the TraceToChain artifact.
#
#   bash reproduce.sh --quick      # ~2 min   smoke test (reduced sample sizes)
#   bash reproduce.sh              # ~25 min  synthetic + MAST studies (SS1-SS9)
#   bash reproduce.sh --full       # ~2.5 h   everything, incl. real-trace SS10-SS15
#
# Every script writes a JSON summary next to its figure; those JSONs are the
# ground truth for every number quoted in the paper.  All RNGs are seeded
# (see data/synthetic/seeds.json and the SEED constant in each SS1x script).
#
# Prereqs: python >= 3.10 with numpy, scipy, matplotlib, pandas.
#          PRISM 4.7+ on $PATH is optional (SS5 is skipped without it).
#          No GPU, no network, no proprietary data.

set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

MODE="standard"; QUICK=""
case "${1-}" in
  --quick) MODE="quick"; QUICK="--quick" ;;
  --full)  MODE="full" ;;
  "")      ;;
  *) echo "usage: bash reproduce.sh [--quick|--full]" >&2; exit 2 ;;
esac
echo "[reproduce] mode: $MODE"

run() {  # run <label> <script> [args...]
  local label="$1"; shift
  printf '[reproduce] %-42s' "$label"
  local t0=$SECONDS
  if python3 "$@" >"/tmp/tracetochain_$(basename "$1" .py).log" 2>&1; then
    echo "ok ($((SECONDS-t0))s)"
  else
    echo "FAILED -- see /tmp/tracetochain_$(basename "$1" .py).log"; return 1
  fi
}

# --quick uses reduced sample sizes, so its outputs do NOT match the paper.
# Snapshot the archived results and restore them afterwards, so a smoke test
# can never leave the artifact in a state that disagrees with the manuscript.
SNAP=""
if [ "$MODE" = "quick" ]; then
  SNAP="$(mktemp -d)"
  cp -R data figs "$SNAP"/
  # shellcheck disable=SC2064
  trap "rm -rf data figs; cp -R '$SNAP'/data '$SNAP'/figs .; rm -rf '$SNAP'; \
        echo '[reproduce] archived results restored (quick mode is a smoke test only)'" EXIT
  echo "[reproduce] quick mode: archived data/ and figs/ will be restored on exit"
else
  echo "[reproduce] NOTE: this regenerates data/ and figs/ in place;"
  echo "[reproduce]       use 'git diff --stat data/' to compare against the shipped results"
fi

echo "[reproduce] (1/4) installing the mcr package"
pip install -e ./code --break-system-packages >/dev/null 2>&1 \
  || pip install -e ./code >/dev/null 2>&1 \
  || echo "[reproduce]   (editable install skipped; assuming mcr is importable)"

cd code

echo "[reproduce] (2/4) analytic checks on synthetic chains (SS1-SS5)"
run "SS1  closed form vs. Monte Carlo"        sim/SS1_cf_vs_mc.py      $QUICK
run "SS2  perturbation bound tightness"       sim/SS2_perturbation.py  $QUICK
run "SS3  correlated-trial (Jensen) gap"      sim/SS3_correlation.py   $QUICK
run "SS4  NHPP rare-failure limit"            sim/SS4_nhpp_limit.py    $QUICK
python3 sim/SS5_prism_cross.py >/dev/null 2>&1 \
  && echo "[reproduce] SS5  PRISM cross-check                    ok" \
  || echo "[reproduce] SS5  PRISM cross-check                    skipped (PRISM not on PATH)"

echo "[reproduce] (3/4) MAST-derived studies (SS6-SS9)"
run "SS6  MAST reliability ranking"           sim/SS6_mast_case_study.py
run "SS7  composite AIC/KS goodness-of-fit"   sim/SS7_goodness_of_fit.py
run "SS7b cross-benchmark archetypes"         sim/SS7_cross_benchmark.py
run "SS8  Dirichlet + bootstrap intervals"    sim/SS8_uncertainty_quantification.py $QUICK
run "SS9  held-out fit/test validation"       sim/SS9_heldout_empirical.py          $QUICK

if [ "$MODE" = "full" ]; then
  echo "[reproduce] (4/4) real agent-trace studies (SS10-SS15)"
  echo "[reproduce]        note: SS11 fits 18 clustered chains and takes ~75 min"
  run "SS10 real SWE-bench fit + certificate"  sim/SS10_swebench_real.py --limit 160
  run "SS12 censoring sensitivity"             sim/SS12_censoring.py
  run "SS13 bootstrap-path comparison"         sim/SS13_bootstrap_paths.py
  run "SS14 order-k lifts on the full corpus"  sim/SS14_higher_order.py
  run "SS15 tau-bench cross-benchmark"         sim/SS15_taubench.py
  run "SS11 clustering ablation (18 configs)"  sim/SS11_ablation.py
else
  echo "[reproduce] (4/4) real agent-trace studies (SS10-SS15)  skipped -- use --full"
fi

cd ..
echo ""
echo "[reproduce] DONE ($MODE)."
echo "   figures   : figs/*.pdf"
echo "   summaries : data/synthetic/*.json  data/mast_derived/*.json"
[ "$MODE" = "full" ] && echo "               data/swebench_real/*.json  data/tau_bench/*.json"
echo "   paper     : cd paper && latexmk -pdf main.tex"
