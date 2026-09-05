"""Diagnostic: why does Fisher's feature-overlap advantage shrink/reverse
at high n_classes (as seen in run_experiment.py's n_classes=5 rows)?

Two confounds are entangled in run_one_cell's n_informative = max(3,
n_classes): (a) more classes -> less redundant/correlated feature budget
(n_redundant = n_features - n_informative shrinks), and (b) more classes ->
fewer hard-labeled perturbation samples per class for Fisher's pooled S_W
and class means, at a fixed N_PERTURB_SAMPLES. This script decouples them:
n_informative is fixed at 3 regardless of n_classes (so redundant budget is
constant), and N_PERTURB_SAMPLES is varied to test the sample-size
hypothesis directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_onevsrest_lasso, fit_fisher, fit_fisher_soft, top_k_indices  # noqa: E402
from metrics import mean_pairwise_feature_overlap  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402

N_FEATURES = 14
N_INFORMATIVE = 3  # fixed regardless of n_classes -> redundant budget constant
K = 6
N_INSTANCES = 8
N_CLASSES_GRID = [3, 4, 5, 6, 7]
N_PERTURB_GRID = [300, 900]
SEED = 0


def run_diagnostic(n_classes, n_perturb, rng):
    n_redundant = N_FEATURES - N_INFORMATIVE
    X, y = make_classification(
        n_samples=2000, n_features=N_FEATURES, n_informative=N_INFORMATIVE,
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

    ovr_overlaps, fisher_overlaps, fisher_soft_overlaps = [], [], []
    min_class_counts, n_present_hard, n_present_soft = [], [], []
    for x in instances:
        Z, weights = sample_perturbations(x, feature_std, n_perturb, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)
        counts = np.array([(hard_labels == c).sum() for c in classes])
        min_class_counts.append(counts.min())

        ovr_lasso = fit_onevsrest_lasso(Z, weights, proba, x, K)
        ovr_topk = {c: ovr_lasso[c]["selected"] for c in range(n_classes)}

        fisher = fit_fisher(Z, weights, hard_labels, classes)
        fisher_vecs = {}
        for c in classes:
            v = fisher["onevsrest_direction"](c)
            if v is not None:
                fisher_vecs[c] = v
        fisher_topk = {c: top_k_indices(v, K) for c, v in fisher_vecs.items()}
        n_present_hard.append(len(fisher_vecs))

        fisher_soft = fit_fisher_soft(Z, weights, proba, classes)
        fisher_soft_vecs = {}
        for c in classes:
            v = fisher_soft["onevsrest_direction"](c)
            if v is not None:
                fisher_soft_vecs[c] = v
        fisher_soft_topk = {c: top_k_indices(v, K) for c, v in fisher_soft_vecs.items()}
        n_present_soft.append(len(fisher_soft_vecs))

        ovr_overlaps.append(mean_pairwise_feature_overlap(ovr_topk))
        fisher_overlaps.append(mean_pairwise_feature_overlap(fisher_topk))
        fisher_soft_overlaps.append(mean_pairwise_feature_overlap(fisher_soft_topk))

    return dict(
        n_classes=n_classes, n_perturb=n_perturb,
        ovr_overlap=float(np.mean(ovr_overlaps)),
        fisher_overlap=float(np.mean(fisher_overlaps)),
        fisher_soft_overlap=float(np.mean(fisher_soft_overlaps)),
        min_class_count_avg=float(np.mean(min_class_counts)),
        n_present_hard_avg=float(np.mean(n_present_hard)),
        n_present_soft_avg=float(np.mean(n_present_soft)),
    )


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for n_classes in N_CLASSES_GRID:
        for n_perturb in N_PERTURB_GRID:
            print(f"running n_classes={n_classes}, n_perturb={n_perturb} ...")
            rows.append(run_diagnostic(n_classes, n_perturb, rng))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    print()
    print(df.to_string(index=False))

    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "reversal_investigation.csv", index=False)


if __name__ == "__main__":
    main()
