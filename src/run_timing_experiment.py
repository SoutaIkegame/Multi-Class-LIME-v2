"""Measure surrogate fitting time for the OVR/OVO comparison.

The benchmark isolates the fitting step: all methods receive the same
perturbation matrix and black-box probabilities.  Times are reported per
explained instance and aggregated over independent dataset seeds.  OVR is
the current implementation that fits one Lasso surrogate for every class;
the OVO methods fit only the selected top-two class pair.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from surrogates import fit_onevsrest_lasso, fit_contrastive_lasso, fit_ovo_logistic_lasso  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = list(range(3, 11))
K_FRACS = [0.25, 0.5]
N_INSTANCES = 4
N_PERTURB_SAMPLES = 300
N_DATASET_SEEDS = 10
SEED = 20260906


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1000.0


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> list[dict]:
    # Keep the expanded 3--10 class sweep valid even for d=8.
    n_informative = min(n_features, max(3, n_classes))
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=0)
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
        for K in sorted({max(1, round(f * n_features)) for f in K_FRACS}):
            rows.append({
                "n_features": n_features, "n_classes": n_classes, "K": K,
                "ovr_union_ms": timed(lambda: fit_onevsrest_lasso(Z, w, proba, x, K)),
                "contrastive_ms": timed(lambda: fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)),
                "logistic_ms": timed(lambda: fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)),
            })
    return rows


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for nf in N_FEATURES_GRID:
        for nc in N_CLASSES_GRID:
            for seed in range(N_DATASET_SEEDS):
                for row in run_one_cell(nf, nc, rng):
                    row["seed"] = seed
                    rows.append(row)
    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "timing_results.csv", index=False)
    summary = (df.groupby(["n_features", "n_classes", "K"])
                 [["ovr_union_ms", "contrastive_ms", "logistic_ms"]]
                 .agg(["mean", "median", "std"]).reset_index())
    summary.to_csv(out / "timing_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
