"""Proposal C: OVO local logistic (soft-label cross-entropy) vs Contrastive's
Ridge-on-log-odds, on the SAME target logit(q) = log(p_c1/p_c2).

Two black boxes, mirroring run_groundtruth_experiment.py / run_pair_kernel_
experiment.py:
  logistic  true theta_c1-theta_c2 known -> Spearman rank correlation,
            recall@K of the true top-K features (K = round(0.5*n_features))
  rf        pairwise-sign fidelity (overall / extreme / moderate, same
            EXTREME_THRESHOLD split as run_extreme_regime_experiment.py)
            and normalized direction variance under resampling

CAVEAT (found 2026-09-05): the rf-side fidelity/extreme/moderate numbers
are measured on the SAME Z used to fit each variant -- in-sample fit
quality, not evidence of generalization to an unseen neighborhood. The
logistic-side Spearman/recall numbers are not affected by this (they
compare fitted coefficients to the true population theta, not to
predictions on the fitting sample). The held-out re-measurement of the rf
comparison lives in run_combined_bc_experiment.py's *_test columns --
treat that script's numbers as authoritative for C's fidelity claims, not
this one's.

Usage: python3 src/run_logistic_target_experiment.py
Output: results/logistic_target_{logistic,rf}_{results,stats}.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_contrastive, fit_ovo_logistic, top_k_indices  # noqa: E402
from metrics import pairwise_coef_spearman, total_variance_normalized  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
N_STABILITY_REPEATS = 8
EXTREME_THRESHOLD = 0.15
SEED = 0
N_DATASET_SEEDS = 20


def _sign_acc(coef, b, Z, proba, c1, c2, w, mask=None):
    pred = np.sign(Z @ coef + b)
    true = np.sign(proba[:, c1] - proba[:, c2])
    if mask is None:
        mask = np.ones(len(w), dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(pred[mask] == true[mask], weights=w[mask]))


def run_one_cell_rf(n_features, n_classes, rng):
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        extreme = np.minimum(proba[:, c1], proba[:, c2]) < EXTREME_THRESHOLD

        fits = {
            "ridge": fit_contrastive(Z, w, proba, c1, c2, x),
            "logistic": fit_ovo_logistic(Z, w, proba, c1, c2, x),
        }
        row = dict(n_features=n_features, n_classes=n_classes,
                   frac_extreme=float(np.average(extreme, weights=w)))
        for m, f in fits.items():
            row[f"{m}_fidelity"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w)
            row[f"{m}_extreme"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, extreme)
            row[f"{m}_moderate"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, ~extreme)

        hist = {m: [fits[m]["coef"]] for m in fits}
        for _ in range(N_STABILITY_REPEATS - 1):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            pk = clf.predict_proba(Zk)
            hist["ridge"].append(fit_contrastive(Zk, wk, pk, c1, c2, x)["coef"])
            hist["logistic"].append(fit_ovo_logistic(Zk, wk, pk, c1, c2, x)["coef"])
        for m in fits:
            row[f"{m}_direction_variance"] = total_variance_normalized(hist[m])
        rows.append(row)
    return rows


def run_one_cell_logistic(n_features, n_classes, rng):
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = LR(max_iter=2000, C=1.0).fit(X_train, y_train)
    theta = clf.coef_
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)
    K = max(1, round(0.5 * n_features))

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])
        beta_true = theta[c1] - theta[c2]
        true_top = top_k_indices(beta_true, K)

        Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        fits = {
            "ridge": fit_contrastive(Z, w, proba, c1, c2, x),
            "logistic": fit_ovo_logistic(Z, w, proba, c1, c2, x),
        }
        row = dict(n_features=n_features, n_classes=n_classes)
        for m, f in fits.items():
            row[f"{m}_spearman"] = pairwise_coef_spearman(beta_true, f["coef"])
            row[f"{m}_recall"] = len(top_k_indices(f["coef"], K) & true_top) / len(true_top)
        rows.append(row)
    return rows


def run(black_box: str, out: Path):
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    fn = run_one_cell_rf if black_box == "rf" else run_one_cell_logistic
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{black_box}] [{time.time()-t0:6.1f}s] n_features={n_features}, n_classes={n_classes}", flush=True)
            for seed in range(N_DATASET_SEEDS):
                rows = fn(n_features, n_classes, rng)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    df.to_csv(out / f"logistic_target_{black_box}_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in df.columns if c not in ("n_features", "n_classes", "seed")]
    print(f"\n=== [{black_box}] overall ===")
    print(df[cols].mean().round(4))

    metrics = ["fidelity", "extreme", "moderate", "direction_variance"] if black_box == "rf" else ["spearman", "recall"]
    pairs = [(m, f"logistic_{m}", f"ridge_{m}") for m in metrics]
    stats = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats.to_csv(out / f"logistic_target_{black_box}_stats.csv", index=False)
    print(f"\n=== [{black_box}] paired tests (20 seeds, Holm within metric); mean_diff = logistic - ridge ===")
    print(stats[["n_features", "n_classes", "metric", "mean_a", "mean_b", "mean_diff",
                 "p_value", "effect_size", "p_value_holm_reject"]].to_string(index=False))


def main():
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["logistic", "rf"]
    for bb in which:
        run(bb, out)


if __name__ == "__main__":
    main()
