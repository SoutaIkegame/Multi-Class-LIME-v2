"""Fidelity: how well does each surrogate approximate the black box's
actual predicted probabilities over the local perturbed neighborhood?

one-vs-rest directly regresses on f_c(z), so it optimizes this by
construction. Fisher LDA does not natively output probabilities -- it
optimizes class separation, not fit to f(z) -- so to compare fairly we
convert its centroids/S_W into the standard LDA probabilistic model
(Gaussian class-conditional density with shared covariance S_W, i.e.
exactly what scikit-learn's LinearDiscriminantAnalysis.predict_proba does):

    log P(c | z) ~ -0.5 (z-mu_c)^T S_W^{-1} (z-mu_c) + log(prior_c)

normalized via softmax over the classes present in the local neighborhood.

Loss: weighted squared Hellinger distance (same choice as SLISEMAP, Eq. 11:
bounded in [0,1], numerically stable, symmetric -- unlike KL). Weighted by
the same LIME kernel pi_i used everywhere else in this codebase.
"""
from __future__ import annotations

import numpy as np


def onevsrest_predict_proba(ovr_fit: dict, Z: np.ndarray, n_classes: int) -> np.ndarray:
    """Raw regression outputs, clipped to be non-negative and renormalized
    to sum to 1 per row -- this is what you'd have to do in practice to
    turn LIME's regression outputs into a valid probability vector, and is
    itself evidence of the LIMEtree critique (p.5): "modeling probabilities
    with linear regression... may be given a numerical prediction outside
    of this range [0,1]"."""
    raw = np.zeros((Z.shape[0], n_classes))
    for c in range(n_classes):
        raw[:, c] = Z @ ovr_fit[c]["coef"] + ovr_fit[c]["intercept"]
    clipped = np.clip(raw, 0.0, None)
    row_sums = clipped.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return clipped / row_sums


def fisher_predict_proba(fisher_fit: dict, Z: np.ndarray, priors: dict, n_classes: int) -> np.ndarray:
    """LDA-style probabilistic prediction from a fit_fisher / fit_fisher_soft
    result: Gaussian class-conditional density with the pooled S_W as the
    shared covariance, softmax-normalized. Classes absent from the local
    neighborhood (mu missing) get a floor probability rather than being
    silently dropped, so the comparison to the black box's full n_classes
    output is well-defined."""
    mu = fisher_fit["mu"]
    S_W_inv = fisher_fit["S_W_inv"]
    present = sorted(mu.keys())
    n_present = len(present)

    scores = np.full((Z.shape[0], n_present), -np.inf)
    for j, c in enumerate(present):
        diff = Z - mu[c][None, :]
        maha = np.einsum("ij,jk,ik->i", diff, S_W_inv, diff)
        log_prior = np.log(max(priors.get(c, 1e-6), 1e-12))
        scores[:, j] = -0.5 * maha + log_prior

    scores -= scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    proba_present = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    proba = np.full((Z.shape[0], n_classes), 1e-6)
    for j, c in enumerate(present):
        proba[:, c] = proba_present[:, j]
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


def hard_label_priors(hard_labels: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> dict:
    priors = {}
    total = weights.sum()
    for c in classes:
        mask = hard_labels == c
        priors[c] = float(weights[mask].sum() / total) if total > 0 else 0.0
    return priors


def soft_label_priors(proba: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> dict:
    priors = {}
    total = weights.sum()
    for idx, c in enumerate(classes):
        priors[c] = float((weights * proba[:, idx]).sum() / total) if total > 0 else 0.0
    return priors


def weighted_hellinger_loss(proba_pred: np.ndarray, proba_true: np.ndarray, weights: np.ndarray) -> float:
    per_point = 1.0 - np.sum(np.sqrt(np.clip(proba_pred, 0, None) * np.clip(proba_true, 0, None)), axis=1)
    return float(np.average(per_point, weights=weights))
