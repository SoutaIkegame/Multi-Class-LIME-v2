"""Compare the four exact-K explanations across black-box structures.

The experiment separates three cases:

* ``linear_softmax``: multinomial logistic regression. Pairwise log-odds is
  exactly linear in the original features.
* ``nonlinear_softmax``: an MLP whose final multiclass probabilities use a
  softmax, but whose logits are nonlinear functions of the input.
* ``random_forest``: a probability-producing black box without softmax logits.

All four explanation methods use the same selected top-1/top-2 pair, LIME
neighborhood, proximity weights, held-out neighborhood, and exact number K of
displayed features.  This tests whether a benefit is tied to pair selection,
to a softmax output, or specifically to linear logits.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from run_ovo_vs_ovr_experiment import (  # noqa: E402
    _pair_q,
    _ridge_refit,
    _sign_acc,
    _soft_logistic_refit,
    _weighted_brier,
)
from stats_utils import compare_methods  # noqa: E402
from surrogates import (  # noqa: E402
    fit_contrastive_lasso,
    fit_onevsrest_lasso,
    fit_ovo_logistic_lasso,
    fit_pairwise_lime_lasso,
)


BLACKBOX_TYPES = ["linear_softmax", "nonlinear_softmax", "random_forest"]
N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 5]
K_FRACS = [0.25, 0.5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
N_DATASET_SEEDS = 20
SEED = 20260909


def _make_blackbox(kind: str, random_state: int):
    if kind == "linear_softmax":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)
    if kind == "nonlinear_softmax":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16), activation="tanh", solver="adam",
                alpha=1e-3, batch_size=64, learning_rate_init=3e-3,
                max_iter=400, early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=20, random_state=random_state,
            ),
        )
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=random_state)
    raise ValueError(f"unknown black-box type: {kind}")


def _exact_ovr(Z, w, proba, x, c1, c2, K):
    selectors = fit_onevsrest_lasso(Z, w, proba, x, K)
    first1 = _ridge_refit(Z, w, proba[:, c1], selectors[c1]["selected"], x)
    first2 = _ridge_refit(Z, w, proba[:, c2], selectors[c2]["selected"], x)
    candidate = np.array(sorted(first1["selected"] | first2["selected"]), dtype=int)
    difference = first1["coef"] - first2["coef"]
    selected = frozenset(candidate[np.argsort(-np.abs(difference[candidate]))[:K]].tolist())
    final1 = _ridge_refit(Z, w, proba[:, c1], selected, x)
    final2 = _ridge_refit(Z, w, proba[:, c2], selected, x)
    return {
        "coef": final1["coef"] - final2["coef"],
        "intercept": final1["intercept"] - final2["intercept"],
        "selected": selected,
    }


def _explain_one(clf, x, feature_std, rng, K):
    x_proba = clf.predict_proba(x[None, :])[0]
    order = np.argsort(x_proba)[::-1]
    c1, c2 = int(order[0]), int(order[1])
    Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
    Z_test, w_test = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
    proba = clf.predict_proba(Z)
    proba_test = clf.predict_proba(Z_test)

    ovr = _exact_ovr(Z, w, proba, x, c1, c2, K)
    pair_lime = fit_pairwise_lime_lasso(Z, w, proba, c1, c2, x, K)
    con_selector = fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)
    log_selector = fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)

    eps = 1e-6
    q_train = _pair_q(proba, c1, c2, eps)
    true_q_test = _pair_q(proba_test, c1, c2, eps)
    log_ratio = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    con = _ridge_refit(Z, w, log_ratio, con_selector["selected"], x)
    logistic = _soft_logistic_refit(Z, w, q_train, log_selector["selected"], x)

    pair_lime_q = Z_test @ pair_lime["coef"] + pair_lime["intercept"]
    con_q = expit(Z_test @ con["coef"] + con["intercept"])
    logistic_q = expit(Z_test @ logistic["coef"] + logistic["intercept"])
    return {
        "ovr_fidelity_test": _sign_acc(
            ovr["coef"], ovr["intercept"], Z_test, proba_test, c1, c2, w_test),
        "two_class_lime_fidelity_test": _sign_acc(
            pair_lime["coef"], pair_lime["intercept"] - 0.5,
            Z_test, proba_test, c1, c2, w_test),
        "contrastive_fidelity_test": _sign_acc(
            con["coef"], con["intercept"], Z_test, proba_test, c1, c2, w_test),
        "logistic_fidelity_test": _sign_acc(
            logistic["coef"], logistic["intercept"], Z_test, proba_test, c1, c2, w_test),
        "two_class_lime_brier_test": _weighted_brier(pair_lime_q, true_q_test, w_test),
        "contrastive_brier_test": _weighted_brier(con_q, true_q_test, w_test),
        "logistic_brier_test": _weighted_brier(logistic_q, true_q_test, w_test),
        "ovr_complexity": len(ovr["selected"]),
        "two_class_lime_complexity": len(pair_lime["selected"]),
        "contrastive_complexity": len(con["selected"]),
        "logistic_complexity": len(logistic["selected"]),
    }


def run_one_dataset(n_features, n_classes, dataset_seed):
    n_informative = min(n_features, max(3, n_classes))
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_repeated=0, n_classes=n_classes,
        n_clusters_per_class=1, class_sep=1.2, random_state=dataset_seed,
    )
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    k_values = sorted({max(1, round(f * n_features)) for f in K_FRACS})
    rows = []
    for kind_i, kind in enumerate(BLACKBOX_TYPES):
        clf = _make_blackbox(kind, dataset_seed + 1009 * kind_i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X_train, y_train)
        instances = pick_contested_instances(clf, X_test, N_INSTANCES)
        rng = np.random.default_rng(dataset_seed + 100_003 * (kind_i + 1))
        for x in instances:
            for K in k_values:
                row = _explain_one(clf, x, feature_std, rng, K)
                row.update(blackbox=kind, n_features=n_features,
                           n_classes=n_classes, K=K)
                rows.append(row)
    return rows


def main():
    rows = []
    t0 = time.time()
    cells = [(nf, nc) for nf in N_FEATURES_GRID for nc in N_CLASSES_GRID]
    for cell_i, (nf, nc) in enumerate(cells, 1):
        print(f"[{time.time()-t0:6.1f}s] cell {cell_i}/{len(cells)}: "
              f"d={nf}, C={nc}, {N_DATASET_SEEDS} seeds", flush=True)
        for seed_i in range(N_DATASET_SEEDS):
            dataset_seed = SEED + 10_000 * cell_i + seed_i
            batch = run_one_dataset(nf, nc, dataset_seed)
            for row in batch:
                row["seed"] = seed_i
            rows.extend(batch)

    df = pd.DataFrame(rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "blackbox_comparison_results.csv", index=False)

    pairs = [
        ("fidelity", "two_class_lime_fidelity_test", "ovr_fidelity_test"),
        ("fidelity", "contrastive_fidelity_test", "ovr_fidelity_test"),
        ("fidelity", "logistic_fidelity_test", "ovr_fidelity_test"),
        ("fidelity", "contrastive_fidelity_test", "two_class_lime_fidelity_test"),
        ("fidelity", "logistic_fidelity_test", "two_class_lime_fidelity_test"),
        ("fidelity", "logistic_fidelity_test", "contrastive_fidelity_test"),
        ("brier", "contrastive_brier_test", "two_class_lime_brier_test"),
        ("brier", "logistic_brier_test", "two_class_lime_brier_test"),
        ("brier", "logistic_brier_test", "contrastive_brier_test"),
    ]
    stats_parts = []
    for kind in BLACKBOX_TYPES:
        part = compare_methods(
            df[df.blackbox == kind], ["n_features", "n_classes", "K"], pairs)
        part.insert(0, "blackbox", kind)
        stats_parts.append(part)
    stats = pd.concat(stats_parts, ignore_index=True)
    stats.to_csv(out / "blackbox_comparison_stats.csv", index=False)

    metric_cols = [
        "ovr_fidelity_test", "two_class_lime_fidelity_test",
        "contrastive_fidelity_test", "logistic_fidelity_test",
        "two_class_lime_brier_test", "contrastive_brier_test",
        "logistic_brier_test",
    ]
    print("\n=== overall by black-box ===")
    print(df.groupby("blackbox")[metric_cols].mean().round(5).to_string())
    print("\n=== Holm-significant cells by black-box and pair ===")
    print(stats.groupby(["blackbox", "metric", "method_a", "method_b"])
          .p_value_holm_reject.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
