"""Harder ground-truth benchmark for pairwise multiclass LIME variants.

Unlike ``run_groundtruth_experiment.py``, the pairwise log-ratio is not a
globally linear target.  Each independently drawn black box has correlated
inputs, weak dense nuisance effects, class-shared effects, pair-specific
effects, nonlinear curvature, and additional classes.  The reference answer
at an explained point is the numerical gradient of log(p_A / p_B).

All explainers receive the same Top-A/Top-B point, perturbations, proximity
weights, held-out perturbations, and exact display budget K.  We report:

* recall of the true local gradient's top-K features;
* cosine similarity to the full true local gradient;
* held-out A-vs-B sign fidelity.

The experiment is a controlled synthetic benchmark, but it is deliberately
not an identity test: the target is curved and the truth changes with x.
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
from run_blackbox_comparison_experiment import _exact_ovr  # noqa: E402
from run_ovo_vs_ovr_experiment import (  # noqa: E402
    _pair_q,
    _ridge_refit,
    _sign_acc,
    _soft_logistic_refit,
)
from stats_utils import compare_methods  # noqa: E402
from surrogates import (  # noqa: E402
    fit_contrastive_lasso,
    fit_ovo_logistic_lasso,
    fit_pairwise_lime_lasso,
)


N_FEATURES = 30
N_CLASSES_GRID = [3, 5, 8]
CURVATURE_GRID = [0.35, 0.80]
K = 5
N_INSTANCES = 6
N_PERTURB_SAMPLES = 400
N_DATASET_SEEDS = 20
SEED = 20260910


class CurvedPairBlackBox:
    """Smooth multiclass softmax model with a known, nonconstant gradient."""

    def __init__(self, n_classes: int, curvature: float, rng: np.random.Generator):
        self.n_classes = n_classes
        self.curvature = curvature
        perm = rng.permutation(N_FEATURES)
        self.shared_idx = perm[:6]
        self.pair_idx = perm[6:14]
        self.other_idx = perm[14:]

        self.shared_w = rng.normal(0.0, 1.0, 6)
        self.pair_w = rng.normal(0.0, 1.0, 8)
        self.pair_w *= np.linspace(1.35, 0.55, 8)
        self.leak = rng.uniform(0.06, 0.16)

        # Weak dense terms ensure that the true local ranking is not simply a
        # fixed list of the eight planted pair features.
        self.dense_w = rng.normal(0.0, 0.075, (n_classes, N_FEATURES))
        self.other_w = rng.normal(0.0, 0.55, (max(0, n_classes - 2), len(self.other_idx)))
        self.phase = rng.uniform(-np.pi, np.pi, n_classes)
        self.intercept = np.full(n_classes, -0.45)
        self.intercept[:2] = [0.30, 0.27]

    def _shared(self, X: np.ndarray) -> np.ndarray:
        V = X[:, self.shared_idx]
        linear = V @ self.shared_w
        curved = 0.45 * np.sin(V[:, 0] + 0.6 * V[:, 1]) + 0.18 * V[:, 2] * V[:, 3]
        return 0.85 * linear + self.curvature * curved

    def _pair(self, X: np.ndarray) -> np.ndarray:
        V = X[:, self.pair_idx]
        linear = V @ self.pair_w
        curved = (
            0.65 * np.sin(V[:, 0] * V[:, 1])
            + 0.30 * V[:, 2] * V[:, 3]
            + 0.16 * (V[:, 4] ** 2 - V[:, 5] ** 2)
        )
        return 0.72 * linear + self.curvature * curved

    def scores(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        S = X @ self.dense_w.T + self.intercept
        shared = self._shared(X)
        pair = self._pair(X)
        S[:, 0] += (1.0 + self.leak) * shared + pair
        S[:, 1] += (1.0 - self.leak) * shared - pair
        for c in range(2, self.n_classes):
            V = X[:, self.other_idx]
            S[:, c] += 0.52 * (V @ self.other_w[c - 2])
            S[:, c] += self.curvature * 0.35 * np.sin(V[:, c % len(self.other_idx)] + self.phase[c])
        return S

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.scores(X), axis=1)

    def pair_logratio(self, X: np.ndarray) -> np.ndarray:
        S = self.scores(X)
        return S[:, 0] - S[:, 1]

    def pair_gradient(self, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
        # Central difference on log(p_A/p_B)=score_A-score_B.  This also
        # checks the actual implemented black-box function instead of relying
        # on a separately coded symbolic gradient.
        eye = np.eye(N_FEATURES) * h
        plus = self.pair_logratio(x[None, :] + eye)
        minus = self.pair_logratio(x[None, :] - eye)
        return (plus - minus) / (2.0 * h)


def correlated_candidates(rng: np.random.Generator, n: int = 50_000) -> np.ndarray:
    rho = 0.55
    idx = np.arange(N_FEATURES)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    return rng.multivariate_normal(np.zeros(N_FEATURES), cov, size=n)


def pick_ab_contested(bb: CurvedPairBlackBox, rng: np.random.Generator) -> np.ndarray:
    X = correlated_candidates(rng)
    p = bb.predict_proba(X)
    top2 = np.sort(np.argsort(p, axis=1)[:, -2:], axis=1)
    eligible = np.flatnonzero(np.all(top2 == np.array([0, 1]), axis=1))
    if len(eligible) < N_INSTANCES:
        raise RuntimeError(f"only {len(eligible)} A/B contested candidates")
    # This matches the intended application: explain points where A and B are
    # the two leading competitors.  Avoid selecting only the six most exact
    # ties by sampling across the lowest 10% of eligible margins.
    margin = np.abs(p[eligible, 0] - p[eligible, 1])
    pool_n = max(N_INSTANCES, int(0.10 * len(eligible)))
    pool = eligible[np.argsort(margin)[:pool_n]]
    return X[rng.choice(pool, size=N_INSTANCES, replace=False)]


def topk_recall(selected: frozenset[int], truth: np.ndarray) -> float:
    true_top = set(np.argsort(-np.abs(truth))[:K].tolist())
    return len(set(selected) & true_top) / K


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else float("nan")


def explain_one(bb: CurvedPairBlackBox, x: np.ndarray, rng: np.random.Generator) -> dict:
    std = np.ones(N_FEATURES)
    Z, w = sample_perturbations(x, std, N_PERTURB_SAMPLES, rng)
    Z_test, w_test = sample_perturbations(x, std, N_PERTURB_SAMPLES, rng)
    proba = bb.predict_proba(Z)
    proba_test = bb.predict_proba(Z_test)
    c1, c2 = 0, 1

    ovr = _exact_ovr(Z, w, proba, x, c1, c2, K)
    pair = fit_pairwise_lime_lasso(Z, w, proba, c1, c2, x, K)
    con_sel = fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)
    log_sel = fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)

    eps = 1e-6
    q = _pair_q(proba, c1, c2, eps)
    log_ratio = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
    con = _ridge_refit(Z, w, log_ratio, con_sel["selected"], x)
    logistic = _soft_logistic_refit(Z, w, q, log_sel["selected"], x)
    truth = bb.pair_gradient(x)

    models = {
        "ovr": (ovr, ovr["intercept"]),
        "two_class_lime": (pair, pair["intercept"] - 0.5),
        "contrastive": (con, con["intercept"]),
        "logistic": (logistic, logistic["intercept"]),
    }
    row = {}
    for name, (model, boundary) in models.items():
        row[f"{name}_topk_recall"] = topk_recall(model["selected"], truth)
        row[f"{name}_cosine"] = cosine(model["coef"], truth)
        row[f"{name}_fidelity"] = _sign_acc(
            model["coef"], boundary, Z_test, proba_test, c1, c2, w_test
        )
    return row


def main() -> None:
    rows = []
    t0 = time.time()
    cells = [(c, curv) for c in N_CLASSES_GRID for curv in CURVATURE_GRID]
    for cell_i, (n_classes, curvature) in enumerate(cells, 1):
        print(f"[{time.time()-t0:6.1f}s] cell {cell_i}/{len(cells)}: "
              f"C={n_classes}, curvature={curvature}, seeds={N_DATASET_SEEDS}", flush=True)
        for seed in range(N_DATASET_SEEDS):
            rng = np.random.default_rng(SEED + 100_000 * cell_i + seed)
            bb = CurvedPairBlackBox(n_classes, curvature, rng)
            for x in pick_ab_contested(bb, rng):
                row = explain_one(bb, x, rng)
                row.update(n_classes=n_classes, curvature=curvature, K=K, seed=seed)
                rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results"
    df.to_csv(out / "challenging_groundtruth_results.csv", index=False)

    pairs = []
    for metric in ["topk_recall", "cosine", "fidelity"]:
        for method in ["two_class_lime", "contrastive", "logistic"]:
            pairs.append((metric, f"{method}_{metric}", f"ovr_{metric}"))
    stats = compare_methods(df, ["n_classes", "curvature", "K"], pairs)
    stats.to_csv(out / "challenging_groundtruth_stats.csv", index=False)

    cols = [c for c in df if c.endswith(("topk_recall", "cosine", "fidelity"))]
    print("\n=== mean by curvature and classes ===")
    print(df.groupby(["curvature", "n_classes"])[cols].mean().round(4).to_string())
    print("\n=== overall ===")
    print(df[cols].mean().round(4).to_string())
    print("\n=== Holm-significant cells ===")
    print(stats.groupby(["metric", "method_a", "method_b"])
          .p_value_holm_reject.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
