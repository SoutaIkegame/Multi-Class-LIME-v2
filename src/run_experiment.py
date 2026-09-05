"""Main experiment: one-vs-rest LIME vs Fisher LIME, compared on the
literature-grounded definitions of the multi-class problem:

  (1) structural consistency -- LIMEtree (Sokol & Flach 2025, Sec. 3, p.5)
      names the actual failure mode as per-class surrogates that "do not
      share a common tree structure or split on different feature subsets".
      We operationalize this for linear surrogates as: take each class's
      top-K explanation features (as a real sparse LIME explanation would
      display, LIME's own 'highest_weights' selection mode), then measure
      the average pairwise Jaccard overlap of these top-K feature sets
      across all classes. High overlap = shared structure; low overlap =
      LIMEtree's named failure mode.
  (2) sum-to-one deviation under that SAME top-K truncation -- shows how
      the structural divergence in (1) is what actually degrades
      sum-to-one (not independent fitting per se, see README/results notes).
  (3) stability -- variance of the competing-pair direction vector under
      repeated perturbation resampling (Q1-3 from the original discussion).

We removed the "transitivity violation rate" metric used in an earlier
version of this script: it is provably always 0 for ANY method that
assigns one real-valued score per class (a >b, b>c => a>c holds for any
three real numbers, regardless of how they were computed), so it cannot
discriminate between methods. See the conversation record / commit history
for the derivation.

Usage: python3 src/run_experiment.py
Output: results/experiment_results.csv (+ printed summary)
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
from surrogates import fit_onevsrest, fit_onevsrest_lasso, fit_fisher, top_k_indices  # noqa: E402
from metrics import (  # noqa: E402
    sum_to_one_deviation,
    sum_to_one_deviation_topk,
    mean_pairwise_feature_overlap,
    total_variance,
    total_variance_normalized,
    mean_norm,
)
from stats_utils import compare_methods  # noqa: E402

# n_features grid chosen to always leave room for redundant (correlated)
# features on top of the informative core: n_informative = max(3, n_classes),
# n_redundant = n_features - n_informative.
N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
K_FRACS = [0.3, 0.6]  # top-K as a fraction of n_features ("sparse" / "less sparse")
N_INSTANCES = 8
N_STABILITY_REPEATS = 15
N_PERTURB_SAMPLES = 300
SEED = 0
# Each grid cell is re-run over this many INDEPENDENT dataset draws
# (fresh make_classification + fresh RandomForest fit each time) so that
# grid-cell means can be reported with a real confidence interval and
# method-vs-method differences can be tested for significance instead of
# eyeballed off a single dataset draw. See src/stats_utils.py docstring.
N_DATASET_SEEDS = 20


def pick_contested_instances(clf, X_pool, n_instances):
    proba = clf.predict_proba(X_pool)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    order = np.argsort(margin)  # smallest margin first: most contested points
    idx = order[:n_instances]
    return X_pool[idx]


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> list[dict]:
    # n_informative "core" features determine the classes; n_redundant are
    # random linear combinations of the core features, i.e. deliberately
    # correlated with them -- the multicollinearity that makes independent
    # per-class Lasso feature selection unstable/arbitrary (a well-known
    # Lasso property), which is what we want to stress-test here.
    n_informative = max(3, n_classes)
    n_redundant = max(0, n_features - n_informative)
    X, y = make_classification(
        n_samples=2000,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)

    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)
    k_values = sorted({max(1, round(frac * n_features)) for frac in K_FRACS})

    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        # --- one shared-sampling run for the structural-consistency metrics ---
        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba, x)
        fisher = fit_fisher(Z, weights, hard_labels, classes)

        local_preds = {c: ovr[c]["local_pred"] for c in range(n_classes)}
        sum_dev = sum_to_one_deviation(local_preds)

        fisher_ovr_vecs = {c: fisher["onevsrest_direction"](c) for c in classes}
        fisher_ovr_vecs = {c: v for c, v in fisher_ovr_vecs.items() if v is not None}

        # --- competing pair: predicted class vs runner-up, by black box ---
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])

        # --- repeated resampling for the stability metric ---
        ovr_diffs, fisher_diffs = [], []
        for k in range(N_STABILITY_REPEATS):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            probak = clf.predict_proba(Zk)
            hard_k = probak.argmax(axis=1)

            ovr_k = fit_onevsrest(Zk, wk, probak, x)
            fisher_k = fit_fisher(Zk, wk, hard_k, classes)

            ovr_diffs.append(ovr_k[c_star]["coef"] - ovr_k[c_runner]["coef"])
            v_k = fisher_k["pairwise_direction"](c_star, c_runner)
            if v_k is not None:
                fisher_diffs.append(v_k)

        # Raw-scale variance is NOT comparable across methods: Fisher's
        # v = S_W^{-1}(mu_X-mu_Y) has an arbitrary scale set by S_W's
        # magnitude, unlike one-vs-rest's regression coefficients. Report
        # the scale-invariant (unit-norm) version as the headline stability
        # metric, and the raw version + typical norms alongside it for
        # transparency about the scale gap.
        ovr_var = total_variance(ovr_diffs)
        fisher_var = total_variance(fisher_diffs) if len(fisher_diffs) >= 2 else float("nan")
        ovr_var_norm = total_variance_normalized(ovr_diffs)
        fisher_var_norm = total_variance_normalized(fisher_diffs) if len(fisher_diffs) >= 2 else float("nan")
        ovr_mean_norm = mean_norm(ovr_diffs)
        fisher_mean_norm = mean_norm(fisher_diffs)

        for K in k_values:
            # Lasso-path-style selection: each class's K features are chosen
            # independently by its own regularization path, so which K
            # features "win" among the correlated group can genuinely differ
            # across classes -- unlike Ridge-then-truncate, which mostly just
            # re-ranks the same globally-informative features.
            ovr_lasso = fit_onevsrest_lasso(Z, weights, proba, x, K)
            ovr_topk = {c: ovr_lasso[c]["selected"] for c in range(n_classes)}
            ovr_lasso_intercepts = {c: ovr_lasso[c]["intercept"] for c in range(n_classes)}
            ovr_lasso_coefs = {c: ovr_lasso[c]["coef"] for c in range(n_classes)}

            fisher_topk = {c: top_k_indices(v, K) for c, v in fisher_ovr_vecs.items()}

            ovr_overlap = mean_pairwise_feature_overlap(ovr_topk)
            fisher_overlap = mean_pairwise_feature_overlap(fisher_topk)

            ovr_sum_dev_topk = sum_to_one_deviation_topk(
                ovr_lasso_intercepts, ovr_lasso_coefs, ovr_topk, x)

            rows.append(dict(
                n_features=n_features, n_classes=n_classes, K=K,
                sum_to_one_deviation=sum_dev,
                sum_to_one_deviation_topk=ovr_sum_dev_topk,
                ovr_feature_overlap=ovr_overlap,
                fisher_feature_overlap=fisher_overlap,
                ovr_pairdiff_variance=ovr_var,
                fisher_pairdiff_variance=fisher_var,
                ovr_pairdiff_variance_normalized=ovr_var_norm,
                fisher_pairdiff_variance_normalized=fisher_var_norm,
                ovr_pairdiff_mean_norm=ovr_mean_norm,
                fisher_pairdiff_mean_norm=fisher_mean_norm,
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
                # run_one_cell draws a fresh make_classification dataset (and
                # fits a fresh RandomForest) each call, consuming from rng --
                # so each seed here is a genuinely independent replicate, not
                # just a different perturbation draw on the same dataset.
                rows = run_one_cell(n_features, n_classes, rng)
                for row in rows:
                    row["seed"] = seed
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "experiment_results.csv", index=False)

    summary = df.groupby(["n_features", "n_classes", "K"]).mean(numeric_only=True)
    summary.to_csv(out_dir / "experiment_summary.csv")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-(n_features, n_classes, K) mean over instances/repeats/seeds (naive, no CI) ===")
    print(summary)

    print("\n=== overall mean across all grid cells ===")
    print(df.mean(numeric_only=True))

    # --- statistically rigorous comparison: seed-level means, bootstrap
    # CIs, paired Wilcoxon tests, Holm-Bonferroni correction across cells ---
    topk_pairs = [
        ("feature_overlap", "ovr_feature_overlap", "fisher_feature_overlap"),
    ]
    stats_topk = compare_methods(df, ["n_features", "n_classes", "K"], topk_pairs)

    nontopk_pairs = [
        ("stability_normalized", "ovr_pairdiff_variance_normalized", "fisher_pairdiff_variance_normalized"),
    ]
    stats_nontopk = compare_methods(df, ["n_features", "n_classes"], nontopk_pairs)

    stats_all = pd.concat([stats_topk, stats_nontopk], ignore_index=True)
    stats_all.to_csv(out_dir / "experiment_stats.csv", index=False)

    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(Holm-Bonferroni corrected across grid cells within each metric) ===")
    print(stats_all.to_string(index=False))


if __name__ == "__main__":
    main()
