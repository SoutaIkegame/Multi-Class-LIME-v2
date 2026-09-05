"""Sanity checks for algebraic identities that MUST hold given the fitting
methods used, independent of any black box or data draw. These are not
empirical claims to be tested with statistics -- they are consequences of
Ridge regression being a linear operator on its target for a fixed design
matrix, weights, and regularization. If either check fails, something in
the pipeline (weights, design matrix, or regularization) is NOT actually
shared the way the rest of this codebase assumes it is, which would
invalidate comparisons that rely on that assumption.

1. OVR-difference identity: fitting Ridge(p_c) and Ridge(p_d) independently
   (same Z, weights, alpha) and subtracting the coefficients is
   algebraically identical to fitting Ridge directly on (p_c - p_d).
   beta_hat = (Z^T W Z + lambda I)^{-1} Z^T W y is linear in y, so
   Ridge(y1) - Ridge(y2) = Ridge(y1 - y2) whenever Z, W, lambda match.
   This means the "one-vs-rest LIME, then take the coefficient difference"
   baseline used throughout this codebase is NOT a distinct method from
   "directly regress p_c - p_d" -- they are the same estimator. The real
   change Contrastive LIME introduces is the TARGET TRANSFORM (probability
   difference -> log-ratio), not "direct fitting" vs "fit-then-subtract".

2. Cycle consistency for dense (non-Lasso) Contrastive: with the same Z,
   weights, and alpha shared across all three pairs, log-ratio Ridge
   satisfies beta_ab + beta_bc = beta_ac exactly, again by linearity of
   Ridge in its target: log(p_a/p_b) + log(p_b/p_c) = log(p_a/p_c) is an
   exact identity on the TARGET, and Ridge fitting preserves linear
   relationships among targets fit on the same (Z, weights, alpha). This
   holds automatically -- it is not "nothing forces it" (as an earlier,
   now-corrected docstring suggested) but a mathematical guarantee, for
   the dense fit only. fit_contrastive_lasso can break this: each pair's
   sparsest-Lasso-with->=K search can select a different feature subset
   and effectively a different regularization, so the shared-(Z,weights,
   alpha) precondition no longer holds.

Usage: python3 src/check_identities.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_onevsrest, fit_contrastive  # noqa: E402


def check_smoothing_consistency(tol=1e-9) -> bool:
    """fit_contrastive's target log((p1+eps)/(p2+eps)) and fit_ovo_logistic's
    q = (p1+eps)/(p1+p2+2*eps) must satisfy logit(q) == that log-ratio
    EXACTLY for every row, independent of any model fitting -- this is a
    check on the smoothing convention itself, not on any estimator."""
    rng = np.random.default_rng(2)
    p1 = rng.uniform(0, 1, size=1000)
    p2 = rng.uniform(0, 1 - p1)
    eps = 1e-6
    log_ratio = np.log((p1 + eps) / (p2 + eps))
    q = (p1 + eps) / (p1 + p2 + 2 * eps)
    logit_q = np.log(q / (1 - q))
    err = float(np.max(np.abs(log_ratio - logit_q)))
    ok = err < tol
    print(f"[eps-smoothing consistency] max err={err:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_ovr_difference_identity(tol=1e-9) -> bool:
    rng = np.random.default_rng(0)
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                                n_classes=4, n_clusters_per_class=1, random_state=0)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
    feature_std = X_train.std(axis=0)
    x = X_test[0]
    Z, w = sample_perturbations(x, feature_std, 300, rng)
    proba = clf.predict_proba(Z)
    c1, c2 = 0, 1

    ovr = fit_onevsrest(Z, w, proba, x)
    diff_coef = ovr[c1]["coef"] - ovr[c2]["coef"]
    diff_intercept = ovr[c1]["intercept"] - ovr[c2]["intercept"]

    direct = Ridge(alpha=1.0).fit(Z, proba[:, c1] - proba[:, c2], sample_weight=w)

    coef_err = float(np.max(np.abs(diff_coef - direct.coef_)))
    intercept_err = float(abs(diff_intercept - direct.intercept_))
    ok = coef_err < tol and intercept_err < tol
    print(f"[OVR-difference identity] coef max err={coef_err:.2e}, "
          f"intercept err={intercept_err:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_cycle_consistency(tol=1e-9) -> bool:
    rng = np.random.default_rng(1)
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                                n_classes=4, n_clusters_per_class=1, random_state=1)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
    feature_std = X_train.std(axis=0)
    x = X_test[0]
    Z, w = sample_perturbations(x, feature_std, 300, rng)
    proba = clf.predict_proba(Z)
    a, b, c = 0, 1, 2

    fit_ab = fit_contrastive(Z, w, proba, a, b, x)
    fit_bc = fit_contrastive(Z, w, proba, b, c, x)
    fit_ac = fit_contrastive(Z, w, proba, a, c, x)

    coef_err = float(np.max(np.abs((fit_ab["coef"] + fit_bc["coef"]) - fit_ac["coef"])))
    intercept_err = float(abs((fit_ab["intercept"] + fit_bc["intercept"]) - fit_ac["intercept"]))
    ok = coef_err < tol and intercept_err < tol
    print(f"[Cycle consistency, dense Contrastive] coef max err={coef_err:.2e}, "
          f"intercept err={intercept_err:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = check_ovr_difference_identity()
    ok2 = check_cycle_consistency()
    ok3 = check_smoothing_consistency()
    sys.exit(0 if (ok1 and ok2 and ok3) else 1)
