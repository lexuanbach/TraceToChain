"""SWE-bench trajectory adapter for TraceToChain (SS10, real-data study).

This adapter converts *real* SWE-agent execution traces (SWE-bench Verified
leaderboard submissions) into the ``mcr.trace_to_chain.TraceStep`` format so
the same audited pipeline used on synthetic corpora (SS9) can be applied to
operational agent traces. It directly addresses the reviewers' central
request: fit at least one real LLM-agent trace corpus end-to-end and report
the goodness-of-fit outcome honestly.

Provenance of the data
----------------------
Traces come from the public, anonymous-readable S3 bucket that backs the
SWE-bench leaderboard (``s3://swe-bench-submissions``; see
``analysis/download_logs.py`` in github.com/SWE-bench/experiments, which
accesses it with ``botocore.UNSIGNED``). For each submission, every task
instance has a ``trajs/<instance>/<instance>.traj`` JSON file containing a
``trajectory`` list of steps; each step has an ``action`` (the command the
agent issued), an ``observation``, the model ``response``/``thought``, and a
``state``. Ground-truth success/failure for each instance is the SWE-bench
``resolved`` flag from the submission's ``results/results.json``.

Because the raw ``.traj`` files are large (often several MB each, dominated by
tool observations), the harvesting step extracts only a compact, canonical
*action token* per step (no observations are stored). The token rule is:

    1. take the first non-empty line of the action string;
    2. strip any leading ``cd <path> &&`` / ``sudo`` / ``source ... &&`` prefix
       (SWE-agent wraps bash commands as ``cd /testbed && <cmd>``);
    3. the command word is the first remaining token;
    4. for the SWE-agent file editor (``str_replace_editor`` / ``edit`` /
       ``insert``) keep the sub-command, yielding ``edit:view``,
       ``edit:create``, ``edit:str_replace``, ``edit:insert``,
       ``edit:undo_edit``;
    5. a null/empty action becomes ``none``.

The harvested corpus is a JSON file::

    {"<instance_id>": {"toks": [tok, tok, ...],
                       "exit": "<exit_status>",
                       "resolved": true/false}, ...}

This module maps each token to a semantic *category*, turns each step into a
feature vector, and yields ``TraceStep`` sequences with a terminal
success/failure absorbing step taken from ``resolved``. The featurizer is a
rule-based map (the same style of rule-based featurizer used in SS7), so the
whole pipeline is reproducible from the harvested corpus.

NOTE: ``submit`` may appear multiple times inside a single trajectory (the
agent attempts to submit, is asked to continue, and proceeds), so it is NOT
treated as the absorbing event. Absorption is the end of the trajectory; the
absorbing label is the SWE-bench ``resolved`` verdict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mcr.trace_to_chain import TraceStep


# --------------------------------------------------------------------------
# Semantic action categories (the agent "state alphabet").
# --------------------------------------------------------------------------
# Fixed order -> one-hot index. Chosen from the observed SWE-agent action
# vocabulary; rare commands fall into OTHER.
CATEGORIES: list[str] = [
    "VIEW",     # inspect a file via the editor (edit:view)
    "EDIT",     # modify code (edit:create / str_replace / insert / undo_edit)
    "EXECUTE",  # run code or tests (python, pytest, tox, ...)
    "SEARCH",   # locate code (grep, find, rg, locate, ag, ack)
    "INSPECT",  # read via shell (cat, head, tail, less, wc, ls, tree)
    "VCS",      # version control (git)
    "SUBMIT",   # the agent's submit action (non-terminal; see module docstring)
    "FILEOP",   # filesystem mutation (rm, cp, mv, mkdir, touch, chmod)
    "NONE",     # null / unparseable action
    "OTHER",    # everything else (pip, conda, sed, export, cd, echo, psql, ...)
]
_CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

_EXECUTE = {"python", "python3", "pytest", "py.test", "tox", "nosetests",
            "unittest", "make"}
_SEARCH = {"grep", "find", "rg", "locate", "ag", "ack", "fgrep", "egrep"}
_INSPECT = {"cat", "head", "tail", "less", "more", "wc", "ls", "tree",
            "file", "diff", "nl"}
_FILEOP = {"rm", "cp", "mv", "mkdir", "touch", "chmod", "chown", "ln", "rmdir"}


def categorize(tok: str) -> str:
    """Map a canonical action token to a semantic category."""
    if tok.startswith("edit:"):
        sub = tok.split(":", 1)[1]
        return "VIEW" if sub == "view" else "EDIT"
    if tok in _EXECUTE:
        return "EXECUTE"
    if tok in _SEARCH:
        return "SEARCH"
    if tok in _INSPECT:
        return "INSPECT"
    if tok == "git":
        return "VCS"
    if tok == "submit":
        return "SUBMIT"
    if tok in _FILEOP:
        return "FILEOP"
    if tok in ("", "none"):
        return "NONE"
    return "OTHER"


def _step_features(tok: str, depth_frac: float, include_depth: bool) -> np.ndarray:
    """Feature vector for one step: one-hot(category) [+ normalized depth]."""
    v = np.zeros(len(CATEGORIES) + (1 if include_depth else 0), dtype=float)
    v[_CAT_INDEX[categorize(tok)]] = 1.0
    if include_depth:
        v[-1] = depth_frac
    return v


@dataclass
class CorpusStats:
    n_instances: int
    n_used: int
    n_success: int
    n_failure: int
    n_steps: int
    dropped_empty: int


def load_corpus(
    path: str | Path,
    *,
    include_depth: bool = True,
    min_steps: int = 2,
    max_steps: int | None = None,
) -> tuple[list[list[TraceStep]], list[str], CorpusStats]:
    """Load a harvested SWE-bench corpus JSON into TraceStep sequences.

    Returns ``(traces, instance_ids, stats)``. Each trace is a list of
    transient ``TraceStep`` (one per agent step) followed by a single terminal
    step whose ``terminal_label`` is ``"success"`` (resolved) or ``"failure"``
    (not resolved).

    ``min_steps`` drops degenerate trajectories with too few steps to carry
    transition information. ``max_steps`` optionally truncates extremely long
    trajectories (None = keep full length).
    """
    raw = json.loads(Path(path).read_text())
    traces: list[list[TraceStep]] = []
    ids: list[str] = []
    n_succ = n_fail = n_steps = dropped = 0
    for inst in sorted(raw):
        rec = raw[inst]
        toks = list(rec.get("toks", []))
        if max_steps is not None:
            toks = toks[:max_steps]
        if len(toks) < min_steps:
            dropped += 1
            continue
        L = len(toks)
        steps: list[TraceStep] = []
        for j, tok in enumerate(toks):
            depth = j / max(L - 1, 1)
            steps.append(TraceStep(features=_step_features(tok, depth, include_depth)))
        label = "success" if rec.get("resolved") else "failure"
        steps.append(TraceStep(features=np.zeros(len(CATEGORIES) + (1 if include_depth else 0)),
                               is_terminal=True, terminal_label=label))
        traces.append(steps)
        ids.append(inst)
        n_steps += L
        if label == "success":
            n_succ += 1
        else:
            n_fail += 1
    stats = CorpusStats(
        n_instances=len(raw), n_used=len(traces),
        n_success=n_succ, n_failure=n_fail, n_steps=n_steps, dropped_empty=dropped,
    )
    return traces, ids, stats


def category_sequence(rec_toks: Sequence[str]) -> list[str]:
    """Convenience: category sequence for one trajectory's tokens."""
    return [categorize(t) for t in rec_toks]
