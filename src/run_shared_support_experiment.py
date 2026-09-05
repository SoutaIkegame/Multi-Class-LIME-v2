"""Proposal evaluation: two-stage "shared support" surrogates vs per-pair
Lasso, at a fixed explanation size K.

Methods (see surrogates.py, "Two-stage shared support" section):
  pair_lasso     per-pair Contrastive Lasso -- each class pair selects its
                 own K features (LIME-like; the LIMEtree failure mode
                 "different feature subsets" is allowed to happen)
  fisher_select  PROPOSAL: one K-feature subset shared by every pair,
                 chosen from soft-label pooled-S_W Fisher directions, then
                 Contrastive ridge refit on it
  ridge_select   CONTROL: same two-stage scheme, subset chosen from dense
                 OVR ridge magnitudes instead of Fisher directions

Two black boxes:
  logistic  multinomial LogisticRegression -> true theta_c*-theta_c' known:
            recall@K of the true top-K features, and Spearman of the sparse
            refit coefficients vs truth
  rf        RandomForest (the setting used everywhere else): pairwise-sign
            fidelity of the (c*, c') surrogate, stability of the selected
            support (mean Jaccard across perturbation resamples) and of the
            direction (unit-vector variance), and cross-pair support overlap
            (mean Jaccard over all class pairs; == 1 by construction for the
            shared methods, so only pair_lasso's value is informative)

Usage: python3 src/run_shared_support_experiment.py
Output: results/shared_support_{logistic,rf}_{results,stats}.csv
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import (  # noqa: E402
    fit_contrastive_lasso, fit_contrastive_on_support, fit_joint_contrastive,
    shared_support_fisher_soft, shared_support_ridge, top_k_indices,
)
from metrics import pairwise_coef_spearman, jaccard, total_variance_normalized  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
K_FRACS = [0.3, 0.6]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
N_STABILITY_REPEATS = 6
SEED = 0
N_DATASET_SEEDS = 20

METHODS = ["pair_lasso", "fisher_select", "ridge_select", "joint_lasso", "joint_refit"]
# joint_lasso  Proposal A: Multi-task Contrastive LIME (L2,1 joint fit of all
#              class log-probabilities), pair coefficients gamma_c - gamma_d
#              taken directly from the joint model
# joint_refit  same learned shared support, Contrastive ridge refit per pair
#              (isolates "joint selection" from "Lasso shrinkage")


def _sign_fidelity(coef, intercept, Z, proba, c1, c2, weights):
    pred = np.sign(Z @ coef + intercept)
    true = np.sign(proba[:, c1] - proba[:, c2])
    return float(np.average(pred == true, weights=weights))


def _fit_all(Z, weights, proba, x, c1, c2, classes, K):
    """Return {method: fit dict with coef/intercept/selected} for the pair."""
    out = {"pair_lasso": fit_contrastive_lasso(Z, weights, proba, c1, c2, x, K)}
    s_f = shared_support_fisher_soft(Z, weights, proba, classes, K)
    out["fisher_select"] = fit_contrastive_on_support(Z, weights, proba, c1, c2, x, s_f)
    s_r = shared_support_ridge(Z, weights, proba, x, K)
    out["ridge_select"] = fit_contrastive_on_support(Z, weights, proba, c1, c2, x, s_r)
    joint = fit_joint_contrastive(Z, weights, proba, x, K)
    out["joint_lasso"] = joint["pair"](c1, c2)
    out["joint_refit"] = fit_contrastive_on_support(Z, weights, proba, c1, c2, x, joint["selected"])
    return out


def _cycle_residual(pair_coefs: dict) -> float:
    """Mean over class triples (a,b,c) of ||b_ab + b_bc - b_ac|| relative to
    the mean norm of the three vectors. Exactly 0 for any method whose pair
    coefficients are gamma_a - gamma_b (joint model) or a fixed linear map of
    the pair target on one shared support (ridge refit); only per-pair
    Lasso, with pair-specific supports and shrinkage, can violate it."""
    def get(a, b):
        if (a, b) in pair_coefs:
            return pair_coefs[(a, b)]
        return -pair_coefs[(b, a)]
    classes = sorted({c for p in pair_coefs for c in p})
    res = []
    for a, b, c in combinations(classes, 3):
        v = get(a, b) + get(b, c) - get(a, c)
        scale = np.mean([np.linalg.norm(get(a, b)), np.linalg.norm(get(b, c)), np.linalg.norm(get(a, c))])
        res.append(np.linalg.norm(v) / (scale + 1e-12))
    return float(np.mean(res)) if res else float("nan")


def run_one_cell(n_features, n_classes, rng, black_box: str) -> list[dict]:
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    if black_box == "logistic":
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
        theta = clf.coef_
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
        theta = None
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)
    k_values = sorted({max(1, round(f * n_features)) for f in K_FRACS})
    all_pairs = list(combinations(range(n_classes), 2))

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)

        for K in k_values:
            fits = _fit_all(Z, weights, proba, x, c1, c2, classes, K)
            row = dict(n_features=n_features, n_classes=n_classes, K=K)

            if black_box == "logistic":
                beta_true = theta[c1] - theta[c2]
                true_top = top_k_indices(beta_true, K)
                for m, f in fits.items():
                    row[f"{m}_recall"] = len(f["selected"] & true_top) / len(true_top)
                    row[f"{m}_spearman"] = pairwise_coef_spearman(beta_true, f["coef"])
            else:
                for m, f in fits.items():
                    row[f"{m}_fidelity"] = _sign_fidelity(f["coef"], f["intercept"], Z, proba, c1, c2, weights)

                # cross-pair overlap and cycle residual: only pair_lasso can
                # differ across pairs / violate the cycle identity
                supports, coefs = {}, {}
                for (a, b) in all_pairs:
                    if (a, b) == (c1, c2):
                        f = fits["pair_lasso"]
                    elif (a, b) == (c2, c1):
                        f = {"selected": fits["pair_lasso"]["selected"], "coef": -fits["pair_lasso"]["coef"]}
                    else:
                        f = fit_contrastive_lasso(Z, weights, proba, a, b, x, K)
                    supports[(a, b)] = f["selected"]
                    coefs[(a, b)] = f["coef"]
                sims = [jaccard(supports[p], supports[q]) for p, q in combinations(supports, 2)]
                row["pair_lasso_crosspair_overlap"] = float(np.mean(sims)) if sims else float("nan")
                row["pair_lasso_cycle_residual"] = _cycle_residual(coefs)
                for m in METHODS[1:]:
                    row[f"{m}_crosspair_overlap"] = 1.0
                    row[f"{m}_cycle_residual"] = 0.0

                # stability under resampling: support Jaccard + direction variance
                sup_hist = {m: [fits[m]["selected"]] for m in METHODS}
                dir_hist = {m: [fits[m]["coef"]] for m in METHODS}
                for _ in range(N_STABILITY_REPEATS - 1):
                    Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
                    pk = clf.predict_proba(Zk)
                    fk = _fit_all(Zk, wk, pk, x, c1, c2, classes, K)
                    for m in METHODS:
                        sup_hist[m].append(fk[m]["selected"])
                        dir_hist[m].append(fk[m]["coef"])
                for m in METHODS:
                    js = [jaccard(a, b) for a, b in combinations(sup_hist[m], 2)]
                    row[f"{m}_support_stability"] = float(np.mean(js))
                    row[f"{m}_direction_variance"] = total_variance_normalized(dir_hist[m])
            rows.append(row)
    return rows


def run_black_box(black_box: str, out: Path):
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{black_box}] [{time.time()-t0:6.1f}s] n_features={n_features}, n_classes={n_classes}", flush=True)
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng, black_box)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / f"shared_support_{black_box}_results.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    metric_cols = [c for c in df.columns if c not in ("n_features", "n_classes", "K", "seed")]
    print(f"\n=== [{black_box}] naive means per (n_features, n_classes, K) ===")
    print(df.groupby(["n_features", "n_classes", "K"])[metric_cols].mean().round(4))
    print(f"\n=== [{black_box}] overall ===")
    print(df[metric_cols].mean().round(4))

    if black_box == "logistic":
        metrics = ["recall", "spearman"]
    else:
        metrics = ["fidelity", "support_stability", "direction_variance"]
    pairs = []
    for met in metrics:
        pairs += [
            (met, f"joint_refit_{met}", f"ridge_select_{met}"),
            (met, f"joint_refit_{met}", f"pair_lasso_{met}"),
            (met, f"joint_lasso_{met}", f"pair_lasso_{met}"),
            (met, f"joint_refit_{met}", f"joint_lasso_{met}"),
            (met, f"fisher_select_{met}", f"ridge_select_{met}"),
            (met, f"ridge_select_{met}", f"pair_lasso_{met}"),
        ]
    stats = compare_methods(df, ["n_features", "n_classes", "K"], pairs)
    stats.to_csv(out / f"shared_support_{black_box}_stats.csv", index=False)
    print(f"\n=== [{black_box}] paired tests (20 seeds, Holm within metric); mean_diff = a - b ===")
    print(stats[["n_features", "n_classes", "K", "metric", "method_a", "method_b",
                 "mean_a", "mean_b", "mean_diff", "p_value", "effect_size",
                 "p_value_holm_reject"]].to_string(index=False))


def main():
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["logistic", "rf"]
    for bb in which:
        run_black_box(bb, out)


if __name__ == "__main__":
    main()
