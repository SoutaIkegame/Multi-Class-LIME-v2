"""Test whether pairwise explanations spend their K features on A-vs-B evidence.

The synthetic linear-softmax black box has three known feature roles:

* pair features have opposite coefficients for classes A and B;
* shared features have identical coefficients for A and B, so they affect
  A/B versus the remaining classes but cancel exactly in log(p_A / p_B);
* other/noise features do not belong in an A-vs-B explanation.

All methods receive the same top-A/top-B points, neighborhoods and exact
feature budget K.  Selection recall therefore measures a concrete aspect of
interpretability: whether the displayed features answer "why A rather than B".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from run_ovo_vs_ovr_experiment import (  # noqa: E402
    _pair_q,
    _ridge_refit,
    _sign_acc,
    _soft_logistic_refit,
)
from stats_utils import compare_methods  # noqa: E402
from surrogates import (  # noqa: E402
    fit_contrastive_lasso,
    fit_onevsrest_lasso,
    fit_ovo_logistic_lasso,
    fit_pairwise_lime_lasso,
)


N_FEATURES = 12
PAIR_FEATURES = frozenset([0, 1, 2])
SHARED_FEATURES = frozenset([3, 4, 5])
K = len(PAIR_FEATURES)
N_CLASSES_GRID = [3, 5]
SHARED_STRENGTH_GRID = [0.0, 1.0, 3.0, 5.0]
N_DATASET_SEEDS = 20
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
SEED = 20260909


class KnownSoftmaxBlackBox:
    def __init__(self, n_classes: int, shared_strength: float, rng: np.random.Generator):
        self.n_classes = n_classes
        self.coef_ = np.zeros((n_classes, N_FEATURES))
        pair = np.array([1.6, -1.2, 0.9])
        shared = shared_strength * np.array([1.2, -1.0, 0.8])
        self.coef_[0, sorted(PAIR_FEATURES)] = pair
        self.coef_[1, sorted(PAIR_FEATURES)] = -pair
        self.coef_[0, sorted(SHARED_FEATURES)] = shared
        self.coef_[1, sorted(SHARED_FEATURES)] = shared

        # Other-class-only signal makes the setting genuinely multiclass while
        # remaining irrelevant to the exact A-vs-B log-ratio direction.
        available = np.arange(6, N_FEATURES)
        for c in range(2, n_classes):
            idx = available[(2 * (c - 2)) % len(available)]
            idx2 = available[(2 * (c - 2) + 1) % len(available)]
            self.coef_[c, [idx, idx2]] = rng.normal(0.0, 1.1, size=2)
        # Keep the remaining classes plausible enough that a feature shared by
        # A and B matters to each OVR probability (A versus all other classes).
        self.intercept_ = np.full(n_classes, 0.2)
        self.intercept_[0] = 0.35
        self.intercept_[1] = 0.30

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(X @ self.coef_.T + self.intercept_, axis=1)


def pick_ab_contested(bb: KnownSoftmaxBlackBox, rng: np.random.Generator) -> np.ndarray:
    candidates = rng.normal(0.0, 0.55, size=(30_000, N_FEATURES))
    p = bb.predict_proba(candidates)
    order = np.argsort(p, axis=1)[:, -2:]
    mask = np.all(np.sort(order, axis=1) == np.array([0, 1]), axis=1)
    eligible = np.flatnonzero(mask)
    if len(eligible) < N_INSTANCES:
        raise RuntimeError("not enough A/B contested points")
    margin = np.abs(p[eligible, 0] - p[eligible, 1])
    return candidates[eligible[np.argsort(margin)[:N_INSTANCES]]]


def role_metrics(selected: frozenset[int]) -> dict[str, float]:
    pair_count = len(selected & PAIR_FEATURES)
    shared_count = len(selected & SHARED_FEATURES)
    return {
        "pair_recall": pair_count / len(PAIR_FEATURES),
        "shared_selection_rate": shared_count / K,
        "other_selection_rate": (K - pair_count - shared_count) / K,
    }


def explain_one(bb, x, rng):
    std = np.ones(N_FEATURES)
    Z, w = sample_perturbations(x, std, N_PERTURB_SAMPLES, rng)
    Z_test, w_test = sample_perturbations(x, std, N_PERTURB_SAMPLES, rng)
    proba = bb.predict_proba(Z)
    proba_test = bb.predict_proba(Z_test)
    order = np.argsort(bb.predict_proba(x[None, :])[0])[::-1]
    c1, c2 = int(order[0]), int(order[1])
    if {c1, c2} != {0, 1}:
        raise AssertionError("explanation point is not an A/B contest")

    selectors = fit_onevsrest_lasso(Z, w, proba, x, K)
    first1 = _ridge_refit(Z, w, proba[:, c1], selectors[c1]["selected"], x)
    first2 = _ridge_refit(Z, w, proba[:, c2], selectors[c2]["selected"], x)
    candidate = np.array(sorted(first1["selected"] | first2["selected"]), dtype=int)
    first_difference = first1["coef"] - first2["coef"]
    exact_selected = frozenset(
        candidate[np.argsort(-np.abs(first_difference[candidate]))[:K]].tolist()
    )
    final1 = _ridge_refit(Z, w, proba[:, c1], exact_selected, x)
    final2 = _ridge_refit(Z, w, proba[:, c2], exact_selected, x)
    ovr = {
        "coef": final1["coef"] - final2["coef"],
        "intercept": final1["intercept"] - final2["intercept"],
        "selected": exact_selected,
    }
    lime = fit_pairwise_lime_lasso(Z, w, proba, c1, c2, x, K)
    con_selector = fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)
    log_selector = fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)
    eps = 1e-6
    q = _pair_q(proba, c1, c2, eps)
    log_ratio = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    con = _ridge_refit(Z, w, log_ratio, con_selector["selected"], x)
    logistic = _soft_logistic_refit(Z, w, q, log_selector["selected"], x)

    models = {
        "ovr": (ovr, ovr["intercept"]),
        "two_class_lime": (lime, lime["intercept"] - 0.5),
        "contrastive": (con, con["intercept"]),
        "logistic": (logistic, logistic["intercept"]),
    }
    row = {}
    # This is what ordinary class-wise LIME displays for the predicted class.
    # It is evaluated for feature role only because one class probability is
    # not itself an A-vs-B decision function.
    for metric, value in role_metrics(selectors[c1]["selected"]).items():
        row[f"ovr_top_class_{metric}"] = value
    for name, (model, boundary_intercept) in models.items():
        row[f"{name}_fidelity_test"] = _sign_acc(
            model["coef"], boundary_intercept, Z_test, proba_test, c1, c2, w_test
        )
        for metric, value in role_metrics(model["selected"]).items():
            row[f"{name}_{metric}"] = value
    return row


def main():
    rows = []
    t0 = time.time()
    cells = [(c, s) for c in N_CLASSES_GRID for s in SHARED_STRENGTH_GRID]
    for cell_i, (n_classes, shared_strength) in enumerate(cells, 1):
        print(
            f"[{time.time()-t0:6.1f}s] cell {cell_i}/{len(cells)}: "
            f"C={n_classes}, shared={shared_strength}, {N_DATASET_SEEDS} seeds",
            flush=True,
        )
        for seed in range(N_DATASET_SEEDS):
            rng = np.random.default_rng(SEED + 10_000 * cell_i + seed)
            bb = KnownSoftmaxBlackBox(n_classes, shared_strength, rng)
            instances = pick_ab_contested(bb, rng)
            for x in instances:
                row = explain_one(bb, x, rng)
                row.update(
                    n_classes=n_classes,
                    shared_strength=shared_strength,
                    K=K,
                    seed=seed,
                )
                rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    df.to_csv(out / "feature_role_results.csv", index=False)
    pairs = []
    for metric in ["pair_recall", "shared_selection_rate", "fidelity_test"]:
        for method in ["two_class_lime", "contrastive", "logistic"]:
            pairs.append((metric, f"{method}_{metric}", f"ovr_{metric}"))
    for method in ["two_class_lime", "contrastive", "logistic", "ovr"]:
        for metric in ["pair_recall", "shared_selection_rate"]:
            pairs.append(
                (f"{metric}_vs_top_class", f"{method}_{metric}",
                 f"ovr_top_class_{metric}")
            )
    stats = compare_methods(df, ["n_classes", "shared_strength", "K"], pairs)
    stats.to_csv(out / "feature_role_stats.csv", index=False)

    metric_cols = [
        c for c in df.columns
        if c.endswith("pair_recall") or c.endswith("shared_selection_rate")
        or c.endswith("fidelity_test")
    ]
    print("\n=== overall by shared strength ===")
    print(df.groupby("shared_strength")[metric_cols].mean().round(4).to_string())
    print("\n=== Holm-significant cells ===")
    print(stats.groupby(["metric", "method_a", "method_b"])
          .p_value_holm_reject.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
