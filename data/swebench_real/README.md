# SS10 real SWE-bench agent-trace corpus

This directory holds the **real** LLM-agent trace corpus used by SS10
(`code/sim/SS10_swebench_real.py`) to fit `TraceToChain` end-to-end and run the
composite goodness-of-fit certificate on operational data.

## Provenance

- **Source:** public reasoning traces archived for a SWE-bench *Verified*
  leaderboard submission, **SWE-agent + Claude-4-Sonnet**
  (`verified/20250522_sweagent_claude-4-sonnet-20250514`).
- **Storage:** the SWE-bench leaderboard backs trajectories with a public,
  anonymous-readable S3 bucket, `s3://swe-bench-submissions` (see
  `analysis/download_logs.py` in `github.com/SWE-bench/experiments`, which
  reads it with `botocore.UNSIGNED`). Each instance has
  `…/trajs/<id>/<id>.traj` (JSON: a `trajectory` list of steps, each with
  `action`, `observation`, `state`).
- **Ground truth:** the per-submission `results/results.json` `resolved` list
  gives the SWE-bench success/failure verdict for each instance.
- **Harvested:** 479 instances (324 resolved / 155 unresolved), 29,504 agent
  steps. (The full submission has 500 instances; ~21 of the longest
  trajectories were not retrieved, so the harvested resolve rate, 67.6%, is
  marginally above the submission's 63%.)

## Files

- `corpus.json` — `{instance_id: {toks:[action_token,…], resolved:bool, label:"S"/"F"}}`.
  Built by `code/sim/_build_corpus.py` from the harvested chunks.
- `harvest_swebench.js` — the browser-console script used to fetch the `.traj`
  files from the public bucket and extract the per-step **action token** for
  each step (the only field used). Run from a page on the bucket origin.
- `chunks.txt`, `chunks2.txt` — intermediate transfer artifact (the harvested
  payload); `_build_corpus.py` reassembles and decodes them into `corpus.json`.
- `SS10_summary.json` / `SS10_summary.tex` — fit + GoF results (auto-generated).

## Action-token rule (featurizer input)

For each step, take the first non-empty line of `action`; strip a leading
`cd <path> &&` / `sudo` / `source … &&` wrapper; the command word is the token,
except the SWE-agent file editor (`str_replace_editor`/`edit`/`insert`) keeps
its sub-command (`edit:view`, `edit:create`, `edit:str_replace`, …). A
null/empty action becomes `none`. `code/sim/swebench_adapter.py` maps these
tokens to semantic categories and to `mcr` `TraceStep` feature vectors.

## Reproduce

```bash
cd code/sim
python3 _build_corpus.py            # chunks.txt(+2) -> corpus.json
python3 SS10_swebench_real.py --limit 160 --kmax 8   # fit + composite GoF
```

## Result (headline)

The composite AIC∧KS certificate **rejects** the first-order absorbing DTMC for
this real corpus: clustering recovers m≈7 interpretable action-states
(silhouette ≈0.80), but AIC prefers a second-order chain
(Δ_AIC ≈ +470…+817 across configurations) and the held-out first-passage KS
test rejects (p < 1e-4; L∞-RDC ≈ 0.26). The rejection is stable across
sub-sample size (120/160) and featurization (with/without a depth feature).
This is the intended behavior of the safeguard: on traces that are not
adequately first-order Markov, the certificate withholds the first-passage
interpretation.
