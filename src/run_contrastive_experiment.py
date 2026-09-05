"""Three-way comparison: one-vs-rest, Fisher (hard-label), and Contrastive
LIME (direct regression of log((p_c1+eps)/(p_c2+eps)) on z, instead of
subtracting two independently-fit one-vs-rest surrogates).

Runs the same three metrics used in prior experiments, all with the
scale-invariant / methodologically-corrected versions:

  (1) fidelity: weighted sign-agreement between the method's pairwise score
      and the black box's actual p_c*(z) - p_c'(z), on the SAME (c*, c')
      pair the black box itself picked at x (matches run_fidelity_experiment's
      corrected "pairwise sign accuracy" check, extended to Contrastive).
  (2) stability: total_variance_normalized (unit-vector direction variance)
      of each method's (c*, c') pairwise vector under repeated perturbation
      resampling -- raw variance is not comparable across methods whose
      outputs live on different natural scales (see metrics.py docstring
      and docs/RECENT_WORK.md).
  (3) feature overlap: Lasso-selected top-K feature sets.
      - one-vs-rest / Fisher: compared across the n_classes per-class
        explanations (as in run_experiment.py / investigate_reversal.py).
      - Contrastive: compared across all C(n_classes,2) pairwise
        explanations, since Contrastive produces one surrogate per PAIR,
        not per class, and nothing forces those pairs to share structure
        (unlike Fisher's shared S_W).

Usage: python3 src/run_contrastive_experiment.py
Output: results/contrastive_results.csv (+ printed summary)
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
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import (  # noqa: E402
    fit_onevsrest, fit_onevsrest_lasso,
    fit_fisher,
    fit_contrastive, fit_contrastive_lasso,
    top_k_indices,
)
from metrics import total_variance_normalized, mean_norm, mean_pairwise_feature_overlap  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
K_FRACS = [0.3, 0.6]
N_INSTANCES = 8
N_STABILITY_REPEATS = 15
N_PERTURB_SAMPLES = 300
SEED = 0
# See src/stats_utils.py docstring: each grid cell is re-run over this many
# independent dataset draws so method comparisons can be tested for
# significance instead of read off a single draw.
N_DATASET_SEEDS = 20


def pairwise_sign_accuracy(pred_scores: np.ndarray, proba: np.ndarray, c1: int, c2: int,
                            weights: np.ndarray) -> float:
    true_sign = np.sign(proba[:, c1] - proba[:, c2])
    pred_sign = np.sign(pred_scores)
    return float(np.average(pred_sign == true_sign, weights=weights))


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
    k_values = sorted({max(1, round(frac * n_features)) for frac in K_FRACS})

    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])

        # --- (1) fidelity: one shared-sampling run ---
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba, x)
        ovr_scores = Z @ (ovr[c_star]["coef"] - ovr[c_runner]["coef"]) + (ovr[c_star]["intercept"] - ovr[c_runner]["intercept"])
        ovr_fid = pairwise_sign_accuracy(ovr_scores, proba, c_star, c_runner, weights)

        fisher = fit_fisher(Z, weights, hard_labels, classes)
        v = fisher["pairwise_direction"](c_star, c_runner)
        if v is not None:
            midpoint = 0.5 * (fisher["mu"][c_star] + fisher["mu"][c_runner])
            fisher_scores = (Z - midpoint[None, :]) @ v
            fisher_fid = pairwise_sign_accuracy(fisher_scores, proba, c_star, c_runner, weights)
        else:
            fisher_fid = float("nan")

        contrastive = fit_contrastive(Z, weights, proba, c_star, c_runner, x)
        contrastive_scores = Z @ contrastive["coef"] + contrastive["intercept"]
        contrastive_fid = pairwise_sign_accuracy(contrastive_scores, proba, c_star, c_runner, weights)

        # --- (2) stability: repeated resampling for the same (c*, c') pair ---
        ovr_diffs, fisher_diffs, contrastive_diffs = [], [], []
        for _ in range(N_STABILITY_REPEATS):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            probak = clf.predict_proba(Zk)
            hard_k = probak.argmax(axis=1)

            ovr_k = fit_onevsrest(Zk, wk, probak, x)
            ovr_diffs.append(ovr_k[c_star]["coef"] - ovr_k[c_runner]["coef"])

            fisher_k = fit_fisher(Zk, wk, hard_k, classes)
            v_k = fisher_k["pairwise_direction"](c_star, c_runner)
            if v_k is not None:
                fisher_diffs.append(v_k)

            contrastive_k = fit_contrastive(Zk, wk, probak, c_star, c_runner, x)
            contrastive_diffs.append(contrastive_k["coef"])

        ovr_var = total_variance_normalized(ovr_diffs)
        fisher_var = total_variance_normalized(fisher_diffs) if len(fisher_diffs) >= 2 else float("nan")
        contrastive_var = total_variance_normalized(contrastive_diffs)
        ovr_norm = mean_norm(ovr_diffs)
        fisher_norm = mean_norm(fisher_diffs)
        contrastive_norm = mean_norm(contrastive_diffs)

        # --- (3) feature overlap ---
        fisher_ovr_vecs = {c: v for c in classes if (v := fisher["onevsrest_direction"](c)) is not None}
        all_pairs = list(combinations(classes, 2))

        for K in k_values:
            ovr_lasso = fit_onevsrest_lasso(Z, weights, proba, x, K)
            ovr_topk = {c: ovr_lasso[c]["selected"] for c in range(n_classes)}
            fisher_topk = {c: top_k_indices(v, K) for c, v in fisher_ovr_vecs.items()}

            contrastive_topk = {}
            for (c1, c2) in all_pairs:
                cl = fit_contrastive_lasso(Z, weights, proba, int(c1), int(c2), x, K)
                contrastive_topk[(c1, c2)] = cl["selected"]

            ovr_overlap = mean_pairwise_feature_overlap(ovr_topk)
            fisher_overlap = mean_pairwise_feature_overlap(fisher_topk)
            contrastive_overlap = mean_pairwise_feature_overlap(contrastive_topk)

            rows.append(dict(
                n_features=n_features, n_classes=n_classes, K=K,
                ovr_fidelity=ovr_fid, fisher_fidelity=fisher_fid, contrastive_fidelity=contrastive_fid,
                ovr_stability_normalized=ovr_var, fisher_stability_normalized=fisher_var,
                contrastive_stability_normalized=contrastive_var,
                ovr_mean_norm=ovr_norm, fisher_mean_norm=fisher_norm, contrastive_mean_norm=contrastive_norm,
                ovr_feature_overlap=ovr_overlap, fisher_feature_overlap=fisher_overlap,
                contrastive_feature_overlap=contrastive_overlap,
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
    df.to_csv(out_dir / "contrastive_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    summary = df.groupby(["n_features", "n_classes", "K"]).mean(numeric_only=True)
    print("\n=== per-(n_features, n_classes, K) mean (naive, no CI) ===")
    print(summary)
    print("\n=== overall mean ===")
    print(df.mean(numeric_only=True))

    # --- statistically rigorous comparison ---
    nontopk_pairs = [
        ("fidelity", "ovr_fidelity", "fisher_fidelity"),
        ("fidelity", "ovr_fidelity", "contrastive_fidelity"),
        ("fidelity", "fisher_fidelity", "contrastive_fidelity"),
        ("stability_normalized", "ovr_stability_normalized", "fisher_stability_normalized"),
        ("stability_normalized", "ovr_stability_normalized", "contrastive_stability_normalized"),
        ("stability_normalized", "fisher_stability_normalized", "contrastive_stability_normalized"),
    ]
    stats_nontopk = compare_methods(df, ["n_features", "n_classes"], nontopk_pairs)

    topk_pairs = [
        ("feature_overlap", "ovr_feature_overlap", "fisher_feature_overlap"),
        ("feature_overlap", "ovr_feature_overlap", "contrastive_feature_overlap"),
        ("feature_overlap", "fisher_feature_overlap", "contrastive_feature_overlap"),
    ]
    stats_topk = compare_methods(df, ["n_features", "n_classes", "K"], topk_pairs)

    stats_all = pd.concat([stats_nontopk, stats_topk], ignore_index=True)
    stats_all.to_csv(out_dir / "contrastive_stats.csv", index=False)

    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(Holm-Bonferroni corrected across grid cells within each metric+pair) ===")
    print(stats_all.to_string(index=False))


if __name__ == "__main__":
    main()
