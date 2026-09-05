"""Consistency and stability metrics for one-vs-rest LIME vs Fisher LIME.

Structural consistency, per LIMEtree's actual definition (Sokol & Flach
2025, Sec. 3, p.5): per-class explanations are "diverse, inconsistent,
competing or contradictory... whenever these models do not share a common
tree structure or split on different feature subsets". For our linear
surrogates this becomes: do the per-class top-K explanation vectors use
overlapping feature subsets? See mean_pairwise_feature_overlap and
sum_to_one_deviation_topk below.

transitivity_violation_rate is kept only as a documented dead end: it is
provably always 0 for ANY method that assigns one real-valued score per
class (a>b, b>c => a>c holds for any three real numbers regardless of how
they were computed), so it cannot discriminate between methods and is no
longer used by run_experiment.py.

Stability (Q1-3 in the discussion):
  - repeat the perturbation-and-fit procedure K times for the same instance,
    for a fixed competitor pair (predicted class c*, runner-up class c').
  - onevsrest diff^(k) = coef_{c*}^(k) - coef_{c'}^(k)   (vector, per repeat)
  - fisher diff^(k)    = v(c*, c')^(k)                    (vector, per repeat)
  - report trace(Cov(diff across k)) for each method: the total variance of
    the pairwise-direction vector under resampled perturbation noise.
"""
from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
from scipy.stats import spearmanr


def sum_to_one_deviation(local_preds: dict) -> float:
    return abs(sum(local_preds.values()) - 1.0)


def transitivity_violation_rate(diff_fn, classes: np.ndarray) -> float:
    """diff_fn(c1, c2) -> signed score or None if unavailable."""
    n_checked = 0
    n_violated = 0
    for x_c, y_c, z_c in permutations(classes, 3):
        dxy = diff_fn(x_c, y_c)
        dyz = diff_fn(y_c, z_c)
        dxz = diff_fn(x_c, z_c)
        if dxy is None or dyz is None or dxz is None:
            continue
        n_checked += 1
        if dxy > 0 and dyz > 0 and not (dxz > 0):
            n_violated += 1
    if n_checked == 0:
        return float("nan")
    return n_violated / n_checked


def total_variance(vectors: list[np.ndarray]) -> float:
    """trace(Cov(vectors)) -- sum of per-coordinate variance across repeats.

    WARNING: this is only a fair comparison between two methods whose
    output vectors live on the same natural scale. Fisher's pairwise
    direction v = S_W^{-1}(mu_X-mu_Y) has an ARBITRARY scale set by the
    magnitude of S_W (there is no canonical normalization -- rescaling S_W
    by any constant rescales v with no change in what it means), unlike
    one-vs-rest's regression coefficients, which are tied to the actual
    prediction units (probability per unit z). Comparing raw variance
    across such differently-scaled vectors is not meaningful on its own:
    since Var scales roughly with squared magnitude, a vector that is
    merely k times smaller in norm -- for no reason related to stability --
    will show ~k^2 times lower raw variance. Use total_variance_normalized
    for a scale-invariant comparison; report this raw version only
    alongside the typical vector norms for context.
    """
    if len(vectors) < 2:
        return float("nan")
    M = np.vstack(vectors)
    return float(np.var(M, axis=0, ddof=1).sum())


def total_variance_normalized(vectors: list[np.ndarray]) -> float:
    """Scale-invariant stability metric: normalize each vector to unit L2
    norm before computing trace(Cov(.)) across repeats. This measures how
    much the *direction* (which is the only part of these vectors that is
    actually interpreted -- relative feature weights and signs) varies
    under resampling, independent of each method's arbitrary output scale.
    Bounded (unit vectors), so directly comparable between methods."""
    unit_vecs = []
    for v in vectors:
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            unit_vecs.append(v / norm)
    if len(unit_vecs) < 2:
        return float("nan")
    M = np.vstack(unit_vecs)
    return float(np.var(M, axis=0, ddof=1).sum())


def mean_norm(vectors: list[np.ndarray]) -> float:
    if not vectors:
        return float("nan")
    return float(np.mean([np.linalg.norm(v) for v in vectors]))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_pairwise_feature_overlap(topk_sets: dict) -> float:
    """LIMEtree's actual consistency claim, operationalized: do the per-class
    explanations rely on the same feature subset? Average Jaccard overlap of
    the top-K feature indices across every pair of classes' explanation
    vectors. High = shared structure (LIMEtree's desideratum); low = the
    'diverse, inconsistent... split on different feature subsets' failure
    mode LIMEtree names (Sec. 3, p.5)."""
    keys = list(topk_sets.keys())
    if len(keys) < 2:
        return float("nan")
    sims = []
    for c1, c2 in combinations(keys, 2):
        sims.append(jaccard(topk_sets[c1], topk_sets[c2]))
    return float(np.mean(sims))


def pairwise_coef_spearman(true_coef: np.ndarray | None, est_coef: np.ndarray | None) -> float:
    """Rahnama et al. (2024)-style ground-truth check: Spearman rank
    correlation between a method's estimated pairwise coefficient vector
    and the TRUE pairwise coefficient vector of a linear (multinomial
    logistic regression) black box. Rank correlation, not raw magnitude,
    because Ridge/LDA shrinkage biases the *scale* of estimated
    coefficients but should not, for a well-behaved method, disturb their
    relative ranking -- see run_groundtruth_experiment.py."""
    if true_coef is None or est_coef is None:
        return float("nan")
    r = spearmanr(true_coef, est_coef).correlation
    return float(r) if r is not None and not np.isnan(r) else float("nan")


def sum_to_one_deviation_topk(intercepts: dict, coefs: dict, topk_sets: dict, x: np.ndarray) -> float:
    """Same sum-to-one check as sum_to_one_deviation, but using only the
    displayed top-K features of each class's explanation (as a real sparse
    LIME explanation would report), to show how sparsification itself
    degrades the sum-to-one property."""
    total = 0.0
    for c in intercepts:
        idx = np.array(sorted(topk_sets[c]), dtype=int)
        pred = intercepts[c] + float(coefs[c][idx] @ x[idx])
        total += pred
    return abs(total - 1.0)
