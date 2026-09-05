"""Fidelity experiment: one-vs-rest (Ridge, full features) vs Fisher-hard
vs Fisher-soft, measured as weighted squared Hellinger distance between the
surrogate's predicted probability vector and the black box's actual
predict_proba, averaged over the local perturbed neighborhood (the same
kernel-weighted neighborhood used everywhere else in this codebase).

This is the evaluation axis we had NOT tested yet: theory predicts Fisher
should lose here, since it optimizes class separation, not fit to f(z).

Usage: python3 src/run_fidelity_experiment.py
Output: results/fidelity_results.csv (+ printed summary)
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
from surrogates import fit_onevsrest, fit_fisher, fit_fisher_soft  # noqa: E402
from fidelity import (  # noqa: E402
    onevsrest_predict_proba,
    fisher_predict_proba,
    hard_label_priors,
    soft_label_priors,
    weighted_hellinger_loss,
)
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
SEED = 0
# See src/stats_utils.py docstring: each grid cell is re-run over this many
# independent dataset draws so method comparisons can be tested for
# significance instead of read off a single draw.
N_DATASET_SEEDS = 20


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
    clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)

    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba_true = clf.predict_proba(Z)
        hard_labels = proba_true.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba_true, x)
        ovr_proba = onevsrest_predict_proba(ovr, Z, n_classes)
        ovr_loss = weighted_hellinger_loss(ovr_proba, proba_true, weights)

        fisher_hard = fit_fisher(Z, weights, hard_labels, classes)
        hard_priors = hard_label_priors(hard_labels, weights, classes)
        fisher_hard_proba = fisher_predict_proba(fisher_hard, Z, hard_priors, n_classes)
        fisher_hard_loss = weighted_hellinger_loss(fisher_hard_proba, proba_true, weights)

        fisher_soft = fit_fisher_soft(Z, weights, proba_true, classes)
        soft_priors = soft_label_priors(proba_true, weights, classes)
        fisher_soft_proba = fisher_predict_proba(fisher_soft, Z, soft_priors, n_classes)
        fisher_soft_loss = weighted_hellinger_loss(fisher_soft_proba, proba_true, weights)

        rows.append(dict(
            n_features=n_features, n_classes=n_classes,
            ovr_hellinger=ovr_loss,
            fisher_hard_hellinger=fisher_hard_loss,
            fisher_soft_hellinger=fisher_soft_loss,
        ))
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    n_cells = len(N_FEATURES_GRID) * len(N_CLASSES_GRID)
    cell_i = 0
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            cell_i += 1
            print(f"[{time.time()-t0:6.1f}s] cell {cell_i}/{n_cells}: n_features={n_features}, "
                  f"n_classes={n_classes}, {N_DATASET_SEEDS} dataset seeds ...")
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng)
                for row in rows:
                    row["seed"] = seed
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "fidelity_results.csv", index=False)

    summary = df.groupby(["n_features", "n_classes"]).mean(numeric_only=True)
    pd.set_option("display.width", 160)
    print("\n=== per-(n_features, n_classes) mean Hellinger loss, lower=better (naive, no CI) ===")
    print(summary)
    print("\n=== overall mean ===")
    print(df.mean(numeric_only=True))

    # --- statistically rigorous comparison ---
    pairs = [
        ("hellinger", "ovr_hellinger", "fisher_hard_hellinger"),
        ("hellinger", "ovr_hellinger", "fisher_soft_hellinger"),
        ("hellinger", "fisher_hard_hellinger", "fisher_soft_hellinger"),
    ]
    stats_df = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats_df.to_csv(out_dir / "fidelity_stats.csv", index=False)

    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(Holm-Bonferroni corrected across grid cells+pairs; mean_diff = a - b, "
          "negative means a has LOWER (better) Hellinger loss) ===")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
