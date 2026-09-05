"""The two surrogate-fitting methods being compared.

Both take the identical perturbed neighborhood (Z, weights) produced by
perturbation.sample_perturbations, and the identical black-box outputs on
that neighborhood. Only the fitting step differs -- this isolates exactly
the thing the research question is about.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression, MultiTaskLasso, Ridge


def fit_onevsrest(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                   alpha: float = 1.0) -> dict:
    """Classic LIME one-vs-rest: one independent weighted ridge regression
    per class, each regressed directly on that class's black-box probability.

    Returns a dict: class_index -> {"coef": w, "local_pred": predicted P(c) at x}
    """
    n_classes = proba.shape[1]
    out = {}
    for c in range(n_classes):
        model = Ridge(alpha=alpha)
        model.fit(Z, proba[:, c], sample_weight=weights)
        local_pred = float(model.predict(x[None, :])[0])
        out[c] = {"coef": model.coef_.copy(), "intercept": float(model.intercept_), "local_pred": local_pred}
    return out


def _sparsest_lasso_with_at_least_k(Z, weights, y, k, lo=1e-5, hi=10.0, iters=15):
    """Binary-search alpha for the sparsest weighted Lasso fit that still has
    >= k nonzero coefficients (nnz is ~monotone non-increasing in alpha)."""
    def fit_at(alpha):
        m = Lasso(alpha=alpha, max_iter=5000)
        m.fit(Z, y, sample_weight=weights)
        nnz = int(np.sum(np.abs(m.coef_) > 1e-10))
        return nnz, m

    n_lo, m_lo = fit_at(lo)
    n_hi, m_hi = fit_at(hi)
    if n_hi >= k:
        return m_hi
    if n_lo < k:
        return m_lo  # even the loosest alpha can't reach k features

    best = m_lo
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n_mid, m_mid = fit_at(mid)
        if n_mid >= k:
            lo, best = mid, m_mid
        else:
            hi = mid
    return best


def fit_onevsrest_lasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                         K: int) -> dict:
    """A closer reimplementation of LIME's actual 'lasso_path'-style feature
    selection: for each class, find the sparsest weighted Lasso fit with at
    least K nonzero features, keep its top-K by magnitude as the SELECTED
    feature set for that class, then read off intercept/coef/local_pred.

    Unlike fit_onevsrest's Ridge-then-truncate ('highest_weights' mode),
    here the K features actually differ in *which ones enter the model* per
    class -- the mechanism LIMEtree (Sec. 3, p.5) names as the source of
    "diverse, inconsistent... explanations": models that "split on
    different feature subsets".
    """
    n_classes = proba.shape[1]
    out = {}
    for c in range(n_classes):
        model = _sparsest_lasso_with_at_least_k(Z, weights, proba[:, c], K)
        coef = model.coef_.copy()
        idx = np.argsort(-np.abs(coef))[:K]
        mask = np.zeros_like(coef, dtype=bool)
        mask[idx] = True
        coef_masked = np.where(mask, coef, 0.0)
        intercept = float(model.intercept_)
        local_pred = float(intercept + coef_masked @ x)
        out[c] = {
            "coef": coef_masked,
            "intercept": intercept,
            "local_pred": local_pred,
            "selected": frozenset(idx.tolist()),
        }
    return out


def fit_contrastive(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                     x: np.ndarray, eps: float = 1e-6, alpha: float = 1.0) -> dict:
    """"Contrastive LIME": regress log((p_c1+eps)/(p_c2+eps)) on z, rather
    than fitting p_c1 and p_c2 separately and subtracting.

    NOTE on what this actually changes (corrected 2026-09-06): Ridge
    regression is a linear operator on its target for a fixed design
    matrix/weights/alpha (beta_hat = (Z^T W Z + aI)^{-1} Z^T W y is linear
    in y), so "fit_onevsrest(c1) - fit_onevsrest(c2)" and "Ridge fit
    directly on p_c1 - p_c2" are the SAME estimator, not two different
    ones (verified to machine precision in check_identities.py). "Direct
    fitting vs fit-then-subtract" is therefore NOT what distinguishes
    Contrastive from an OVR-difference baseline. The actual change is the
    TARGET TRANSFORM: probability difference (p_c1 - p_c2) -> log-ratio
    (log(p_c1/p_c2)). For a softmax black box the log-ratio target equals
    the raw score difference s_c1 - s_c2 exactly, so sign(prediction) ==
    sign(p_c1(z) - p_c2(z)) holds for the TRUE log-ratio; third classes
    cannot distort the c1-vs-c2 comparison the way they can distort the
    magnitude of a raw probability difference. Whether the FITTED
    (necessarily imperfect, locally-linear) approximation preserves that
    sign is an empirical question, not a guarantee -- hence measuring sign
    agreement as a fidelity metric.

    Cycle consistency: with the same Z, weights, and alpha shared across
    all pairs, beta_ab + beta_bc = beta_ac holds EXACTLY for this dense
    fit (a mathematical consequence of Ridge's linearity in the target,
    not something "nothing forces" -- verified in check_identities.py).
    This guarantee is specific to the dense fit: fit_contrastive_lasso can
    break it, because each pair's sparsest-Lasso-with->=K search can pick
    a different feature subset (and effectively a different regularization
    strength) per pair, so the shared-(Z, weights, alpha) precondition no
    longer holds across pairs.
    """
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = Ridge(alpha=alpha)
    model.fit(Z, y, sample_weight=weights)
    local_pred = float(model.predict(x[None, :])[0])
    return {"coef": model.coef_.copy(), "intercept": float(model.intercept_), "local_pred": local_pred}


def _sparsest_logistic_l1_with_at_least_k(Z2, y2, w2, k, lo=1e-3, hi=100.0, iters=15):
    """Binary-search the L1 inverse-regularization strength C for the
    sparsest weighted logistic fit that still has >= k nonzero coefficients
    (mirrors _sparsest_lasso_with_at_least_k; nnz is ~monotone
    non-decreasing in C here since C is inverse regularization)."""
    def fit_at(C):
        m = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=2000)
        m.fit(Z2, y2, sample_weight=w2)
        nnz = int(np.sum(np.abs(m.coef_[0]) > 1e-10))
        return nnz, m

    n_lo, m_lo = fit_at(lo)
    n_hi, m_hi = fit_at(hi)
    if n_hi < k:
        return m_hi  # even the least regularization can't reach k features
    if n_lo >= k:
        return m_lo

    best = m_hi
    for _ in range(iters):
        mid = float(np.sqrt(lo * hi))
        n_mid, m_mid = fit_at(mid)
        if n_mid >= k:
            hi, best = mid, m_mid
        else:
            lo = mid
    return best


def fit_ovo_logistic_lasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                           x: np.ndarray, K: int, eps: float = 1e-6) -> dict:
    """Sparse (top-K feature) version of fit_ovo_logistic, via L1-penalized
    soft-label logistic regression -- mirrors fit_contrastive_lasso's
    methodology (sparsest fit with >= K nonzero coefficients, then keep the
    top-K by magnitude) so the two can be compared at equal K."""
    q = (proba[:, c1] + eps) / (proba[:, c1] + proba[:, c2] + 2 * eps)
    Z2 = np.vstack([Z, Z])
    y2 = np.concatenate([np.ones(len(Z)), np.zeros(len(Z))])
    w2 = np.concatenate([weights * q, weights * (1.0 - q)])
    model = _sparsest_logistic_l1_with_at_least_k(Z2, y2, w2, K)
    coef = model.coef_[0].copy()
    idx = np.argsort(-np.abs(coef))[:K]
    mask = np.zeros_like(coef, dtype=bool)
    mask[idx] = True
    coef_masked = np.where(mask, coef, 0.0)
    intercept = float(model.intercept_[0])
    return {
        "coef": coef_masked,
        "intercept": intercept,
        "local_pred": float(intercept + coef_masked @ x),
        "selected": frozenset(idx.tolist()),
    }


def fit_contrastive_lasso(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                           x: np.ndarray, K: int, eps: float = 1e-6) -> dict:
    """Lasso-selected (top-K feature) version of fit_contrastive, for the
    feature-overlap experiment -- mirrors fit_onevsrest_lasso's methodology
    but on the log-odds-of-the-pair target."""
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = _sparsest_lasso_with_at_least_k(Z, weights, y, K)
    coef = model.coef_.copy()
    idx = np.argsort(-np.abs(coef))[:K]
    mask = np.zeros_like(coef, dtype=bool)
    mask[idx] = True
    coef_masked = np.where(mask, coef, 0.0)
    intercept = float(model.intercept_)
    local_pred = float(intercept + coef_masked @ x)
    return {
        "coef": coef_masked,
        "intercept": intercept,
        "local_pred": local_pred,
        "selected": frozenset(idx.tolist()),
    }


def fit_fisher(Z: np.ndarray, weights: np.ndarray, hard_labels: np.ndarray,
               classes: np.ndarray, shrinkage: float = 1e-3) -> dict:
    """Multiclass Fisher LDA surrogate with a single pooled S_W across all
    classes present in the (hard-labeled) perturbed neighborhood.

    Returns a dict with the pooled S_W^{-1}, per-class weighted centroids,
    a helper to compute pairwise directions v(X, Y) = S_W^{-1}(mu_X-mu_Y),
    and a one-vs-rest-style per-class direction v(c, not-c) = S_W^{-1}(mu_c -
    mu_not_c) -- the Fisher analogue of LIME's "one surrogate per class",
    used to compare feature usage across classes on equal footing.
    """
    n_features = Z.shape[1]
    present = np.unique(hard_labels)

    mu = {}
    S_W = np.zeros((n_features, n_features))
    for c in present:
        mask = hard_labels == c
        w_c = weights[mask]
        Z_c = Z[mask]
        w_sum = w_c.sum()
        if w_sum <= 0:
            continue
        mu_c = (w_c[:, None] * Z_c).sum(axis=0) / w_sum
        mu[c] = mu_c
        dev = Z_c - mu_c[None, :]
        S_W += (w_c[:, None] * dev).T @ dev

    # shrinkage regularization: S_W + eps * trace(S_W)/d * I
    eps = shrinkage * (np.trace(S_W) / n_features if np.trace(S_W) > 0 else 1.0)
    S_W_reg = S_W + eps * np.eye(n_features)
    S_W_inv = np.linalg.inv(S_W_reg)

    def pairwise_direction(c1, c2):
        if c1 not in mu or c2 not in mu:
            return None
        return S_W_inv @ (mu[c1] - mu[c2])

    def onevsrest_direction(c):
        if c not in mu:
            return None
        mask = hard_labels != c
        w_rest = weights[mask]
        if w_rest.sum() <= 0:
            return None
        mu_rest = (w_rest[:, None] * Z[mask]).sum(axis=0) / w_rest.sum()
        return S_W_inv @ (mu[c] - mu_rest)

    return {
        "mu": mu,
        "S_W": S_W_reg,
        "S_W_inv": S_W_inv,
        "present_classes": present,
        "pairwise_direction": pairwise_direction,
        "onevsrest_direction": onevsrest_direction,
    }


def fit_fisher_soft(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                     classes: np.ndarray, shrinkage: float = 1e-3) -> dict:
    """Soft-label variant of fit_fisher: instead of assigning each z_i to a
    single hard argmax class, every class c gets a weight pi_i * f_c(z_i)
    from every sample. This keeps LIME's original spirit of using the
    continuous probabilities rather than discarding them (see the original
    "soft-label extension" idea), and -- the point of this variant -- it
    means a class is never silently missing from the local neighborhood
    just because it was nobody's single most-likely class: as long as
    f_c(z_i) > 0 somewhere in the neighborhood, mu_c and its contribution to
    S_W are still well-defined. This was empirically found to be the actual
    cause of Fisher's hard-label version losing its feature-overlap
    advantage at higher n_classes (classes vanishing from the local
    argmax-labeled neighborhood), not the feature-redundancy budget.
    """
    n_features = Z.shape[1]
    mu = {}
    S_W = np.zeros((n_features, n_features))
    for idx, c in enumerate(classes):
        soft_w = weights * proba[:, idx]
        w_sum = soft_w.sum()
        if w_sum <= 1e-12:
            continue
        mu_c = (soft_w[:, None] * Z).sum(axis=0) / w_sum
        mu[c] = mu_c
        dev = Z - mu_c[None, :]
        S_W += (soft_w[:, None] * dev).T @ dev

    eps = shrinkage * (np.trace(S_W) / n_features if np.trace(S_W) > 0 else 1.0)
    S_W_reg = S_W + eps * np.eye(n_features)
    S_W_inv = np.linalg.inv(S_W_reg)

    def pairwise_direction(c1, c2):
        if c1 not in mu or c2 not in mu:
            return None
        return S_W_inv @ (mu[c1] - mu[c2])

    def onevsrest_direction(c):
        if c not in mu:
            return None
        idx = int(np.where(classes == c)[0][0])
        soft_w_rest = weights * (1.0 - proba[:, idx])
        w_sum = soft_w_rest.sum()
        if w_sum <= 1e-12:
            return None
        mu_rest = (soft_w_rest[:, None] * Z).sum(axis=0) / w_sum
        return S_W_inv @ (mu[c] - mu_rest)

    return {
        "mu": mu,
        "S_W": S_W_reg,
        "S_W_inv": S_W_inv,
        "present_classes": np.array(sorted(mu.keys())),
        "pairwise_direction": pairwise_direction,
        "onevsrest_direction": onevsrest_direction,
    }


def top_k_indices(vec: np.ndarray, k: int) -> frozenset:
    k = min(k, vec.shape[0])
    order = np.argsort(-np.abs(vec))[:k]
    return frozenset(order.tolist())


# ---------------------------------------------------------------------------
# Two-stage "shared support" surrogates (proposal, 2026-09-05)
#
# Diagnosis (src/diagnose_fisher_direction.py): as a COEFFICIENT estimator
# Fisher's S_W^{-1}(mu_c - mu_d) is dominated by direct log-odds regression
# in this setup (this neighborhood sampling, this label weighting, this
# shrinkage) -- not a proven general limitation of Fisher, since the two
# estimators target different quantities (within-class-scatter-scaled
# centroid distance vs. a direct log-odds regression coefficient). But
# pooling S_W across all classes -- the thing that gives Fisher its
# cross-class shared structure -- costs nothing in this setup.
# So the role left for Fisher is STRUCTURE, not coefficients: choose ONE
# feature subset shared by every class/pair (LIMEtree's "common structure"
# desideratum), then let Contrastive regression supply the coefficients.
#
# The honest control is the same two-stage scheme with the shared subset
# chosen from plain OVR ridge magnitudes instead of Fisher directions. If
# that matches Fisher, the Fisher step adds nothing and the thesis should
# say so.
# ---------------------------------------------------------------------------

def _aggregate_support(vectors: list[np.ndarray], K: int) -> frozenset:
    """Shared top-K support from several per-class direction vectors: each
    vector is normalized to unit norm (so no class dominates by scale), the
    absolute values are summed feature-wise, and the K largest are kept."""
    score = np.zeros_like(vectors[0], dtype=float)
    for v in vectors:
        n = np.linalg.norm(v)
        if n > 1e-12:
            score += np.abs(v) / n
    return top_k_indices(score, K)


def shared_support_fisher_soft(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                               classes: np.ndarray, K: int) -> frozenset:
    """Proposal, stage 1: one feature subset for ALL classes, chosen from the
    soft-label pooled-S_W Fisher one-vs-rest directions
    v_c = S_W^{-1}(mu_c - mu_{not c}). Soft labels because hard-label sample
    starvation was the single largest fixable error in the diagnosis."""
    fit = fit_fisher_soft(Z, weights, proba, classes)
    vecs = [v for c in classes if (v := fit["onevsrest_direction"](c)) is not None]
    if not vecs:
        return frozenset(range(min(K, Z.shape[1])))
    return _aggregate_support(vecs, K)


def shared_support_ridge(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                         K: int) -> frozenset:
    """Control for stage 1: same aggregation rule, but the per-class vectors
    are the dense OVR ridge coefficients (fit_onevsrest) instead of Fisher
    directions. Isolates whether the Fisher geometry itself picks a better
    shared subset than ordinary regression magnitudes do."""
    fit = fit_onevsrest(Z, weights, proba, x)
    vecs = [fit[c]["coef"] for c in sorted(fit)]
    return _aggregate_support(vecs, K)


def fit_ovo_logistic(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                     x: np.ndarray, C: float = 1.0, eps: float = 1e-6) -> dict:
    """Proposal C: fit q(z) = p_c1(z) / (p_c1(z)+p_c2(z)) with a WEIGHTED
    SOFT-LABEL logistic regression instead of Contrastive's Ridge-on-
    log-odds. This isolates the LOSS FUNCTION: cross-entropy (bounded
    gradient, saturates gracefully near q=0/1) vs squared error on a
    log-transformed target (unbounded, blows up near q=0/1 -- the
    mechanism behind the "extreme regime" degradation found earlier).

    q uses the SAME additive smoothing convention as fit_contrastive's
    log((p_c1+eps)/(p_c2+eps)): q = (p_c1+eps) / (p_c1+p_c2+2*eps), so that
    logit(q) == log((p_c1+eps)/(p_c2+eps)) EXACTLY (check: 1-q =
    (p_c2+eps)/(p_c1+p_c2+2*eps), so q/(1-q) = (p_c1+eps)/(p_c2+eps)). Using
    an inconsistent smoothing (e.g. a bare 1e-12 added only to the
    denominator, as an earlier version of this function did) would make
    the two methods' targets subtly different quantities near p=0, not
    just two different losses on the same quantity.

    sklearn's LogisticRegression has no continuous-soft-label fit mode, so
    each row is duplicated into a y=1 case with weight pi_i*q_i and a y=0
    case with weight pi_i*(1-q_i); this is the standard soft-label-via-
    case-weights construction and reproduces the weighted soft cross-
    entropy exactly (sum_i pi_i[q_i log sigma + (1-q_i) log(1-sigma)])."""
    q = (proba[:, c1] + eps) / (proba[:, c1] + proba[:, c2] + 2 * eps)
    Z2 = np.vstack([Z, Z])
    y2 = np.concatenate([np.ones(len(Z)), np.zeros(len(Z))])
    w2 = np.concatenate([weights * q, weights * (1.0 - q)])
    model = LogisticRegression(C=C, max_iter=2000)
    model.fit(Z2, y2, sample_weight=w2)
    coef = model.coef_[0].copy()
    intercept = float(model.intercept_[0])
    return {"coef": coef, "intercept": intercept, "local_pred": float(intercept + coef @ x)}


# ---------------------------------------------------------------------------
# Proposal A (2026-09-05): Multi-task Contrastive LIME -- ONE joint model for
# every class pair.
#
# Per-pair Contrastive Lasso fits C(C,2) independent models, so nothing ties
# their supports together (LIMEtree's "different feature subsets" failure)
# and nothing enforces beta_ab + beta_bc = beta_ac. Here all classes' log-
# probabilities are fit at once as a multi-output regression,
#
#     y_c(z) = log p_c(z) - mean_k log p_k(z),   c = 1..C,
#
# with an L2,1 (group-lasso across classes) penalty: a feature is either
# used by EVERY class or by none, so the shared support is learned, not
# imposed afterwards. The pair explanation is gamma_c - gamma_d, hence
# cycle-consistent by construction; centering by the class mean keeps the
# parameterization symmetric (no arbitrary reference class).
# ---------------------------------------------------------------------------

def _weighted_center(Z, Y, weights):
    w = weights / weights.sum()
    z_bar = w @ Z
    y_bar = w @ Y
    sw = np.sqrt(weights)[:, None]
    return sw * (Z - z_bar), sw * (Y - y_bar), z_bar, y_bar


def _mtl_fit(Zc, Yc, alpha):
    m = MultiTaskLasso(alpha=alpha, fit_intercept=False, max_iter=5000)
    m.fit(Zc, Yc)
    W = m.coef_.T  # (n_features, n_classes)
    nnz = int(np.sum(np.linalg.norm(W, axis=1) > 1e-10))
    return nnz, W


def fit_joint_contrastive(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray, x: np.ndarray,
                          K: int, eps: float = 1e-6, lo: float = 1e-4, hi: float = 100.0,
                          iters: int = 15) -> dict:
    """Multi-task Contrastive LIME at explanation size K: binary-search the
    L2,1 penalty for the sparsest joint fit with >= K active features, keep
    the top-K rows by norm (mirrors _sparsest_lasso_with_at_least_k so the
    comparison with per-pair Lasso is at equal K).

    Returns gamma (n_features x n_classes, zero off-support), per-class
    intercepts, the shared support, and a helper pair(c1, c2) giving the
    dense coefficient vector and intercept of log(p_c1/p_c2)."""
    Y = np.log(proba + eps)
    Y = Y - Y.mean(axis=1, keepdims=True)
    Zc, Yc, z_bar, y_bar = _weighted_center(Z, Y, weights)

    n_lo, W_lo = _mtl_fit(Zc, Yc, lo)
    n_hi, W_hi = _mtl_fit(Zc, Yc, hi)
    if n_hi >= K:
        W = W_hi
    elif n_lo < K:
        W = W_lo
    else:
        W = W_lo
        a_lo, a_hi = lo, hi
        for _ in range(iters):
            mid = float(np.sqrt(a_lo * a_hi))
            n_mid, W_mid = _mtl_fit(Zc, Yc, mid)
            if n_mid >= K:
                a_lo, W = mid, W_mid
            else:
                a_hi = mid
    row_norm = np.linalg.norm(W, axis=1)
    idx = np.argsort(-row_norm)[:K]
    gamma = np.zeros_like(W)
    gamma[idx] = W[idx]
    intercepts = y_bar - z_bar @ gamma  # (n_classes,)
    support = frozenset(idx.tolist())

    def pair(c1: int, c2: int) -> dict:
        coef = gamma[:, c1] - gamma[:, c2]
        b = float(intercepts[c1] - intercepts[c2])
        return {"coef": coef, "intercept": b, "local_pred": float(b + coef @ x), "selected": support}

    return {"gamma": gamma, "intercepts": intercepts, "selected": support, "pair": pair}


def fit_contrastive_on_support(Z: np.ndarray, weights: np.ndarray, proba: np.ndarray,
                               c1: int, c2: int, x: np.ndarray, support: frozenset,
                               eps: float = 1e-6, alpha: float = 1.0) -> dict:
    """Stage 2: Contrastive log-odds ridge restricted to `support`. Returns a
    dense coef vector (zeros off-support) so it is drop-in comparable with
    fit_contrastive / fit_contrastive_lasso outputs."""
    idx = np.array(sorted(support), dtype=int)
    y = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    model = Ridge(alpha=alpha)
    model.fit(Z[:, idx], y, sample_weight=weights)
    coef = np.zeros(Z.shape[1])
    coef[idx] = model.coef_
    intercept = float(model.intercept_)
    return {
        "coef": coef,
        "intercept": intercept,
        "local_pred": float(intercept + coef @ x),
        "selected": frozenset(idx.tolist()),
    }
