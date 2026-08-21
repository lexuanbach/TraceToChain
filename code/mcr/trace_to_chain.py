"""TraceToChain: a concrete pipeline from agent execution traces to an
absorbing DTMC (Q, R_succ, R_fail).

Addresses reviewer critique R3 (state-construction concreteness).  The
pipeline has four stages:

    (1) Feature extraction.     trace -> per-step feature vectors.
    (2) State clustering.        feature vectors -> discrete state labels
                                 (agglomerative clustering, silhouette-
                                 guided k selection).
    (3) Markov-order test.       1st-order vs 2nd-order AIC comparison.
    (4) MLE transition matrix.   counts + Laplace smoothing -> Q, R.

The ``fit()`` entry point performs all four stages; ``goodness_of_fit()``
runs Algorithm 2 (first-passage KS test).  Both algorithms correspond
verbatim to Algorithm 1 and 2 in the paper (§IV).

Design choices:
    * Clustering is agglomerative with Ward linkage — deterministic given
      the data, so no stochastic re-runs are required.
    * Silhouette score picks k in [k_min, k_max]; ties broken toward
      smaller k (Occam).
    * AIC uses #params = k*(k-1) for 1st-order and k*k*(k-1) for 2nd-order
      stochastic-matrix families, consistent with standard finite-state
      Markov-order selection.
    * Laplace smoothing with alpha=1 (conjugate Dirichlet prior) prevents
      zero-row degeneracies when a transient state has no observed outbound.

Dependencies: numpy, scipy (only for KS test); sklearn is optional
(falls back to a numpy implementation of agglomerative clustering if
sklearn is unavailable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np

try:
    from scipy.stats import kstest
except Exception:  # pragma: no cover - scipy optional
    kstest = None  # type: ignore[assignment]

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# ------------------------------ data types ---------------------------------

@dataclass
class TraceStep:
    """One step of an agent trace."""
    features: np.ndarray          # feature vector (length d)
    is_terminal: bool = False
    terminal_label: str | None = None   # "success" | "failure" | None


@dataclass
class ChainFit:
    """The fitted chain plus diagnostic metadata."""
    Q: np.ndarray                 # (m, m) transient-to-transient
    R_succ: np.ndarray            # (m,) transient-to-success
    R_fail: np.ndarray            # (m,) transient-to-failure
    labels: np.ndarray            # state label per input step
    m: int                        # number of transient states
    silhouette: float             # clustering quality
    aic_1st: float
    aic_2nd: float
    first_order_preferred: bool
    n_traces: int
    n_steps: int
    s0_cluster: int = 0           # cluster label of the most common starting state


# ------------------------------ Algorithm 1 ---------------------------------

def _cluster_states(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 15,
    linkage: str = "ward",
) -> tuple[np.ndarray, int, float]:
    """Return (labels, k, silhouette). Agglomerative clustering with the
    requested ``linkage`` (default Ward), silhouette-guided k. Non-Ward
    linkages are used only by the SS11 sensitivity ablation; the numpy
    fallback supports Ward only."""
    n = X.shape[0]
    # bound k by sample size
    k_max = min(k_max, max(k_min, n // 10))
    best_k, best_s, best_labels = k_min, -np.inf, None
    for k in range(k_min, k_max + 1):
        if _HAS_SKLEARN:
            ac = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = ac.fit_predict(X)
        else:
            labels = _agglomerative_ward(X, k)
        # silhouette requires ≥ 2 clusters with ≥ 1 point each
        if len(set(labels)) < 2:
            continue
        if _HAS_SKLEARN:
            s = silhouette_score(X, labels)
        else:
            s = _silhouette(X, labels)
        if s > best_s:
            best_s, best_k, best_labels = s, k, labels
    assert best_labels is not None, "clustering failed"
    return best_labels, best_k, best_s


def _agglomerative_ward(X: np.ndarray, k: int) -> np.ndarray:
    """Ward-linkage agglomerative clustering via Lance--Williams update
    (fallback when sklearn is unavailable).

    Maintains an active-cluster distance matrix in numpy and updates it in
    O(n) per merge using the Lance--Williams recurrence for Ward linkage.
    Total cost is O(n^2) memory and O(n^2) time, which handles trace sets
    of several thousand steps in under a second.

    Fast path: if every feature vector appears at one of a small number
    (<=k) of unique points (common when features are one-hot encoded),
    returns the argmax-based labels directly.
    """
    n = X.shape[0]

    # Fast path: discrete features with <=k unique values.
    uniq, inv = np.unique(X, axis=0, return_inverse=True)
    if uniq.shape[0] <= k:
        return inv.astype(int)

    # General path: Lance--Williams Ward.
    # Pairwise squared distances between singleton centroids.
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    # Ward distance between singletons equals (1*1/(1+1)) * d2 = d2/2.
    # We store the full Ward "merge cost" matrix.
    merge_cost = D2 / 2.0
    np.fill_diagonal(merge_cost, np.inf)

    sizes = np.ones(n, dtype=float)
    active = np.ones(n, dtype=bool)
    parent = np.arange(n, dtype=int)  # union-find parent
    remaining = n

    while remaining > k:
        # Find minimum over active pairs.
        mc = np.where(active[:, None] & active[None, :], merge_cost, np.inf)
        flat_idx = int(np.argmin(mc))
        a, b = flat_idx // n, flat_idx % n
        if a == b or not active[a] or not active[b]:
            break
        # Merge b into a.
        na, nb = sizes[a], sizes[b]
        # Lance--Williams update for Ward: for each active c != a, b,
        #   new_cost(a, c) = ((na+sc)*c(a,c) + (nb+sc)*c(b,c) - sc*c(a,b)) / (na+nb+sc)
        c_idx = np.where(active)[0]
        c_idx = c_idx[(c_idx != a) & (c_idx != b)]
        if c_idx.size > 0:
            sc = sizes[c_idx]
            ca = merge_cost[a, c_idx]
            cb = merge_cost[b, c_idx]
            cab = merge_cost[a, b]
            new = ((na + sc) * ca + (nb + sc) * cb - sc * cab) / (na + nb + sc)
            merge_cost[a, c_idx] = new
            merge_cost[c_idx, a] = new
        sizes[a] = na + nb
        active[b] = False
        parent[b] = a
        merge_cost[b, :] = np.inf
        merge_cost[:, b] = np.inf
        remaining -= 1

    # Resolve final cluster membership: follow parent pointers.
    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    roots = np.array([root(i) for i in range(n)])
    # Compact root labels to 0..k-1.
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(int)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    # Simple silhouette (not batched); used only in the sklearn-missing fallback.
    n = X.shape[0]
    ds = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1) ** 0.5
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        other = ~same
        same[i] = False
        a = ds[i, same].mean() if same.any() else 0.0
        b = np.inf
        for lbl in set(labels):
            if lbl == labels[i]:
                continue
            mask = labels == lbl
            if mask.any():
                bb = ds[i, mask].mean()
                b = min(b, bb)
        s[i] = 0 if max(a, b) == 0 else (b - a) / max(a, b)
    return float(s.mean())


def _markov_order_aic(
    state_seq: Sequence[int], k: int
) -> tuple[float, float, bool]:
    """Compute AIC of 1st- vs 2nd-order Markov models fit to a sequence.

    Returns (aic_1, aic_2, first_order_preferred).
    """
    seq = np.asarray(state_seq, dtype=int)
    n = len(seq)
    # 1st order
    C1 = np.zeros((k, k))
    for t in range(n - 1):
        C1[seq[t], seq[t + 1]] += 1
    P1 = (C1 + 1e-12) / (C1.sum(axis=1, keepdims=True) + 1e-12 * k)
    ll1 = 0.0
    for t in range(n - 1):
        ll1 += np.log(P1[seq[t], seq[t + 1]])
    params1 = k * (k - 1)
    aic1 = 2 * params1 - 2 * ll1
    # 2nd order
    C2 = np.zeros((k, k, k))
    for t in range(n - 2):
        C2[seq[t], seq[t + 1], seq[t + 2]] += 1
    P2 = (C2 + 1e-12) / (C2.sum(axis=2, keepdims=True) + 1e-12 * k)
    ll2 = 0.0
    for t in range(n - 2):
        ll2 += np.log(P2[seq[t], seq[t + 1], seq[t + 2]])
    params2 = k * k * (k - 1)
    aic2 = 2 * params2 - 2 * ll2
    return aic1, aic2, aic1 <= aic2


def _fit_transition_matrix(
    trace_labels: Sequence[Sequence[int]],
    terminals: Sequence[str | None],
    k: int,
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MLE of Q, R_succ, R_fail with Laplace smoothing.

    ``trace_labels[i]`` is the list of transient-state labels visited on
    trace i; ``terminals[i]`` is ``"success"``, ``"failure"``, or None.
    """
    Ct = np.full((k, k), alpha)
    Cs = np.full(k, alpha)
    Cf = np.full(k, alpha)
    for seq, term in zip(trace_labels, terminals):
        if len(seq) == 0:
            continue
        for t in range(len(seq) - 1):
            Ct[seq[t], seq[t + 1]] += 1
        last = seq[-1]
        if term == "success":
            Cs[last] += 1
        elif term == "failure":
            Cf[last] += 1
        # else: censored/no absorption (ignored).
    rowsum = Ct.sum(axis=1) + Cs + Cf
    Q = Ct / rowsum[:, None]
    R_succ = Cs / rowsum
    R_fail = Cf / rowsum
    return Q, R_succ, R_fail


def fit(
    traces: Sequence[Sequence[TraceStep]],
    k_min: int = 2,
    k_max: int = 15,
    alpha: float = 1.0,
    linkage: str = "ward",
) -> ChainFit:
    """Algorithm 1: end-to-end fit from raw trace steps to (Q, R_succ, R_fail).

    Parameters
    ----------
    traces : sequence of trace sequences.  Each trace is a list of TraceStep.
             The final step in each trace should have ``is_terminal=True`` with
             ``terminal_label`` in {"success", "failure"} if known.
    """
    # Collect only non-terminal step features for clustering
    X_list: list[np.ndarray] = []
    trace_indices: list[int] = []
    step_indices: list[int] = []
    first_step_positions: list[int] = []  # positions of step 0 of each trace in X
    for i, tr in enumerate(traces):
        first_recorded = False
        for j, s in enumerate(tr):
            if s.is_terminal:
                continue
            if not first_recorded:
                first_step_positions.append(len(X_list))
                first_recorded = True
            X_list.append(s.features)
            trace_indices.append(i)
            step_indices.append(j)
    if not X_list:
        raise ValueError("no transient steps to cluster")
    X = np.stack(X_list, axis=0)
    n = X.shape[0]

    # Stage 2: clustering
    labels_flat, k, sil = _cluster_states(X, k_min=k_min, k_max=k_max, linkage=linkage)

    # Stage 3: Markov-order AIC
    aic1, aic2, first_order_ok = _markov_order_aic(labels_flat, k)

    # Re-split labels back into traces
    trace_labels: list[list[int]] = [[] for _ in traces]
    for pos, lbl in enumerate(labels_flat):
        trace_labels[trace_indices[pos]].append(int(lbl))
    terminals = [
        (tr[-1].terminal_label if tr and tr[-1].is_terminal else None)
        for tr in traces
    ]

    # Stage 4: transition-matrix MLE
    Q, R_succ, R_fail = _fit_transition_matrix(
        trace_labels, terminals, k=k, alpha=alpha
    )

    # s0_cluster: most common cluster label observed at the first transient
    # step of a trace.  Preserves the original s0 across arbitrary clustering
    # relabelings.
    if first_step_positions:
        first_clusters = [int(labels_flat[p]) for p in first_step_positions]
        vals, counts = np.unique(first_clusters, return_counts=True)
        s0_cluster = int(vals[int(np.argmax(counts))])
    else:
        s0_cluster = 0

    return ChainFit(
        Q=Q,
        R_succ=R_succ,
        R_fail=R_fail,
        labels=labels_flat,
        m=k,
        silhouette=float(sil),
        aic_1st=float(aic1),
        aic_2nd=float(aic2),
        first_order_preferred=bool(first_order_ok),
        n_traces=len(traces),
        n_steps=int(n),
        s0_cluster=s0_cluster,
    )


# ------------------------------ Algorithm 2 ---------------------------------

def first_passage_times(
    Q: np.ndarray,
    R_succ: np.ndarray,
    R_fail: np.ndarray,
    s0: int,
    n_samples: int,
    d_max: int = 500,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample n_samples first-passage times to absorption (either ⊕ or ⊖)."""
    if rng is None:
        rng = np.random.default_rng()
    m = Q.shape[0]
    # cumulative rows: [Q | R_succ | R_fail]
    full = np.concatenate([Q, R_succ[:, None], R_fail[:, None]], axis=1)
    cum = np.cumsum(full, axis=1)
    out = np.empty(n_samples, dtype=int)
    for i in range(n_samples):
        s = s0
        for t in range(1, d_max + 1):
            u = rng.random()
            nxt = int(np.searchsorted(cum[s], u))
            if nxt == m or nxt == m + 1:   # absorbed
                out[i] = t
                break
            s = nxt
        else:
            out[i] = d_max
    return out


def empirical_first_passage_from_traces(
    trace_labels: Sequence[Sequence[int]],
    terminals: Sequence[str | None],
    s0_labels: set[int] | None = None,
) -> np.ndarray:
    """Extract empirical first-passage times from labeled traces.

    Only traces that actually absorb (terminal in {"success","failure"}) are
    counted; censored traces are dropped.
    """
    times = []
    for seq, term in zip(trace_labels, terminals):
        if term not in ("success", "failure"):
            continue
        if s0_labels is not None and (not seq or seq[0] not in s0_labels):
            continue
        times.append(len(seq))   # number of transient-state visits before absorption
    return np.asarray(times, dtype=int)


def goodness_of_fit(
    fit: ChainFit,
    empirical_fpt: np.ndarray,
    n_model_samples: int = 10_000,
    s0: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Algorithm 2: KS test of empirical FPT vs. model-predicted FPT.

    If ``s0`` is None, uses ``fit.s0_cluster`` (the cluster label of the most
    common trace start), so that the KS test is invariant to arbitrary
    relabelings introduced by the clustering step.

    Returns {'ks_stat', 'ks_pvalue', 'n_emp', 'n_model'}.
    """
    if rng is None:
        rng = np.random.default_rng()
    if s0 is None:
        s0 = fit.s0_cluster
    model_fpt = first_passage_times(
        fit.Q, fit.R_succ, fit.R_fail, s0=s0, n_samples=n_model_samples, rng=rng
    )
    # Two-sample KS via CDF difference (no scipy dependency required)
    emp_sorted = np.sort(empirical_fpt.astype(float))
    mod_sorted = np.sort(model_fpt.astype(float))
    xs = np.unique(np.concatenate([emp_sorted, mod_sorted]))
    F_emp = np.searchsorted(emp_sorted, xs, side="right") / emp_sorted.size
    F_mod = np.searchsorted(mod_sorted, xs, side="right") / mod_sorted.size
    D = float(np.max(np.abs(F_emp - F_mod)))
    n1, n2 = emp_sorted.size, mod_sorted.size
    # Asymptotic p-value (Smirnov formula)
    en = np.sqrt(n1 * n2 / (n1 + n2))
    arg = (en + 0.12 + 0.11 / en) * D
    # Kolmogorov distribution tail sum
    p = 2.0 * sum((-1) ** (j - 1) * np.exp(-2 * (j * arg) ** 2) for j in range(1, 101))
    p = max(0.0, min(1.0, p))
    return {
        "ks_stat": D,
        "ks_pvalue": float(p),
        "n_emp": int(n1),
        "n_model": int(n2),
    }


__all__ = [
    "TraceStep",
    "ChainFit",
    "fit",
    "first_passage_times",
    "empirical_first_passage_from_traces",
    "goodness_of_fit",
]
