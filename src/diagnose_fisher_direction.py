"""Diagnostic: WHY does the Fisher pairwise direction S_W^{-1}(mu_c - mu_d)
recover the true coefficient ranking worse than plain regression?

Uses the multinomial-LogisticRegression black box from
run_groundtruth_experiment.py (true pairwise coefficients theta_c - theta_d
are known exactly) and decomposes the Fisher estimator into its two
ingredients, measuring each variant's Spearman rank correlation with the
truth:

  centroid_hard / centroid_soft   mu_c - mu_d alone (no S_W at all)
  pooled_hard / pooled_soft       current fit_fisher / fit_fisher_soft:
                                  S_W^{-1}(mu_c - mu_d), S_W pooled over ALL
                                  classes in the neighborhood
  pair_hard / pair_soft           "OVO Fisher": S_W built only from the two
                                  classes c, d being compared
  diag_hard / diag_soft           S_W replaced by its diagonal (per-feature
                                  variance rescaling only, no rotation)
  cov_logodds                     "continuous-response Fisher":
                                  S_W(soft)^{-1} Cov_pi(z, log(p_c/p_d)) --
                                  same shared metric, but the "between"
                                  direction is estimated from the continuous
                                  log-odds instead of from centroids
  ovr / contrastive               regression references

Reading the table:
  * centroid_* vs pooled_*  -> does S_W^{-1} help or hurt? (in raw feature
    units it SHOULD help: for an isotropic-in-standardized-space cloud and
    a linear boundary, mu_c - mu_d ~ D theta with D = diag(feature var),
    so S_W^{-1} ~ D^{-1} is exactly what undoes the unit distortion)
  * pair_* vs pooled_*      -> is pooling S_W over all classes (the source
    of Fisher's cross-class shared structure) what costs accuracy?
  * diag_* vs pooled_*      -> is it the off-diagonal (rotation) part of
    S_W^{-1} that hurts, or is the diagonal rescaling already enough?
  * cov_logodds vs pooled_soft -> is the loss from the centroid (moment)
    summary discarding the continuous response, rather than from S_W?
  * hard vs soft everywhere -> how much is sample starvation?

Usage: python3 src/diagnose_fisher_direction.py
Output: results/diagnose_fisher_direction_{results,stats}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_onevsrest, fit_contrastive  # noqa: E402
from metrics import pairwise_coef_spearman  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
SEED = 0
N_DATASET_SEEDS = 20
SHRINKAGE = 1e-3


def _weighted_scatter(Z, w_by_class: dict, n_features: int):
    """Return (mu dict, S_W) for arbitrary per-class sample weights."""
    mu = {}
    S_W = np.zeros((n_features, n_features))
    for c, w in w_by_class.items():
        s = w.sum()
        if s <= 1e-12:
            continue
        m = (w[:, None] * Z).sum(axis=0) / s
        mu[c] = m
        dev = Z - m[None, :]
        S_W += (w[:, None] * dev).T @ dev
    return mu, S_W


def _regularize(S_W, n_features):
    tr = np.trace(S_W)
    eps = SHRINKAGE * (tr / n_features if tr > 0 else 1.0)
    return S_W + eps * np.eye(n_features)


def _directions(Z, weights, w_by_class, c1, c2, n_features):
    """Given per-class weights (hard or soft), return the dict of direction
    variants for pair (c1, c2), or None entries where a class is missing."""
    mu_all, S_pooled = _weighted_scatter(Z, w_by_class, n_features)
    if c1 not in mu_all or c2 not in mu_all:
        return dict(centroid=None, pooled=None, pair=None, diag=None)
    diff = mu_all[c1] - mu_all[c2]

    S_pooled_r = _regularize(S_pooled, n_features)
    pooled = np.linalg.solve(S_pooled_r, diff)

    _, S_pair = _weighted_scatter(Z, {c1: w_by_class[c1], c2: w_by_class[c2]}, n_features)
    pair = np.linalg.solve(_regularize(S_pair, n_features), diff)

    diag = diff / np.diag(S_pooled_r)
    return dict(centroid=diff, pooled=pooled, pair=pair, diag=diag)


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> list[dict]:
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
    theta = clf.coef_
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        beta_true = theta[c1] - theta[c2]

        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard = proba.argmax(axis=1)

        w_hard = {c: weights * (hard == c) for c in classes}
        w_soft = {c: weights * proba[:, c] for c in classes}

        d_hard = _directions(Z, weights, w_hard, c1, c2, n_features)
        d_soft = _directions(Z, weights, w_soft, c1, c2, n_features)

        # continuous-response Fisher: shared soft S_W, "between" direction from
        # the kernel-weighted covariance of z with the pair log-odds
        y_lo = np.log((proba[:, c1] + 1e-6) / (proba[:, c2] + 1e-6))
        wsum = weights.sum()
        z_bar = (weights[:, None] * Z).sum(axis=0) / wsum
        y_bar = (weights * y_lo).sum() / wsum
        cov_zy = ((weights * (y_lo - y_bar))[:, None] * (Z - z_bar[None, :])).sum(axis=0) / wsum
        _, S_soft = _weighted_scatter(Z, w_soft, n_features)
        cov_logodds = np.linalg.solve(_regularize(S_soft, n_features), cov_zy)

        ovr = fit_onevsrest(Z, weights, proba, x)
        beta_ovr = ovr[c1]["coef"] - ovr[c2]["coef"]
        beta_con = fit_contrastive(Z, weights, proba, c1, c2, x)["coef"]

        hard_frac = {c: float(w_hard[c].sum() / wsum) for c in (c1, c2)}
        rows.append(dict(
            n_features=n_features, n_classes=n_classes,
            min_pair_class_frac_hard=min(hard_frac.values()),
            centroid_hard=pairwise_coef_spearman(beta_true, d_hard["centroid"]),
            centroid_soft=pairwise_coef_spearman(beta_true, d_soft["centroid"]),
            pooled_hard=pairwise_coef_spearman(beta_true, d_hard["pooled"]),
            pooled_soft=pairwise_coef_spearman(beta_true, d_soft["pooled"]),
            pair_hard=pairwise_coef_spearman(beta_true, d_hard["pair"]),
            pair_soft=pairwise_coef_spearman(beta_true, d_soft["pair"]),
            diag_hard=pairwise_coef_spearman(beta_true, d_hard["diag"]),
            diag_soft=pairwise_coef_spearman(beta_true, d_soft["diag"]),
            cov_logodds=pairwise_coef_spearman(beta_true, cov_logodds),
            ovr=pairwise_coef_spearman(beta_true, beta_ovr),
            contrastive=pairwise_coef_spearman(beta_true, beta_con),
        ))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{time.time()-t0:6.1f}s] n_features={n_features}, n_classes={n_classes}")
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "diagnose_fisher_direction_results.csv", index=False)

    cols = ["centroid_hard", "centroid_soft", "pooled_hard", "pooled_soft",
            "pair_hard", "pair_soft", "diag_hard", "diag_soft", "cov_logodds",
            "ovr", "contrastive", "min_pair_class_frac_hard"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n=== mean Spearman rho vs true theta_c*-theta_c' (naive means) ===")
    print(df.groupby(["n_features", "n_classes"])[cols].mean().round(4))
    print("\n=== overall ===")
    print(df[cols].mean().round(4))

    pairs = [
        ("S_W helps?", "pooled_hard", "centroid_hard"),
        ("S_W helps?", "pooled_soft", "centroid_soft"),
        ("pooling costs?", "pair_hard", "pooled_hard"),
        ("pooling costs?", "pair_soft", "pooled_soft"),
        ("rotation costs?", "diag_soft", "pooled_soft"),
        ("moment vs response", "cov_logodds", "pooled_soft"),
        ("soft vs hard", "pooled_soft", "pooled_hard"),
        ("gap to contrastive", "contrastive", "cov_logodds"),
        ("gap to contrastive", "contrastive", "pooled_soft"),
    ]
    stats = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats.to_csv(out / "diagnose_fisher_direction_stats.csv", index=False)
    print("\n=== paired tests (20 seeds, Holm within each question); mean_diff = a - b ===")
    print(stats[["n_features", "n_classes", "metric", "method_a", "method_b",
                 "mean_a", "mean_b", "mean_diff", "p_value", "effect_size",
                 "p_value_holm_reject"]].to_string(index=False))


if __name__ == "__main__":
    main()
