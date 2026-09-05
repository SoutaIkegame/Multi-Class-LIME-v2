"""Ground-truth recovery experiment, in the spirit of Rahnama et al. 2024
(Data Mining and Knowledge Discovery), "Can Local Explanation Techniques
Explain Linear Additive Models?": when the black box IS itself a linear
model with KNOWN true feature attributions, does a LIME-style local
surrogate actually recover them?

All prior experiments in this codebase test FIDELITY -- does the surrogate
reproduce the black box's OUTPUT in the local neighborhood? A surrogate can
score well on fidelity while attributing the right output to the wrong
features (e.g. by exploiting correlated redundant features). This
experiment tests something different and stronger: does the surrogate's
coefficient vector rank-correlate with the black box's TRUE coefficient
vector?

Design: the black box is replaced with sklearn's multinomial
LogisticRegression instead of RandomForestClassifier. Its coefficients
theta_c are the EXACT global log-odds gradient for class c (log(p_c/p_d) =
(theta_c - theta_d).z + const, exactly, everywhere -- multinomial logistic
regression is linear in log-odds by construction, so there is no locality
question: the "true local answer" at any point is the same global
theta_c - theta_d). Each method's estimated pairwise coefficient vector at
the black box's predicted top class vs runner-up is compared to
theta_c* - theta_c' via Spearman rank correlation (see
metrics.pairwise_coef_spearman) rather than raw magnitude, since Ridge and
LDA shrinkage bias the SCALE of estimates but should not, for a
well-specified method, disturb the RELATIVE ranking.

Expected asymmetry, which this experiment is designed to surface: one-vs-
rest regresses the raw probability p_c(z), which is a NONLINEAR (softmax)
function of z even when the log-odds is linear, so its local gradient is
not proportional to theta_c - theta_d in general. Contrastive regresses
log(p_c/p_d) directly, which for a softmax black box equals
(theta_c-theta_d).z + const EXACTLY, so it should recover the ranking
best. Fisher's LDA direction S_W^{-1}(mu_c-mu_d) is the Bayes-optimal
direction only under a Gaussian-equal-covariance generative assumption
that need not match a discriminatively-trained logistic regression, so its
recovery quality is an open empirical question this experiment answers.

Usage: python3 src/run_groundtruth_experiment.py
Output: results/groundtruth_results.csv (+ results/groundtruth_stats.csv)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import fit_onevsrest, fit_fisher, fit_fisher_soft, fit_contrastive  # noqa: E402
from metrics import pairwise_coef_spearman  # noqa: E402
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
    # The black box itself IS the ground truth here: a multinomial logistic
    # regression's coef_ rows are the exact, everywhere-valid log-odds
    # gradient per class -- unlike RandomForestClassifier (used everywhere
    # else in this codebase), which has no closed-form "true" local
    # coefficients to check a surrogate against.
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(X_train, y_train)
    theta = clf.coef_  # shape (n_classes, n_features)
    feature_std = X_train.std(axis=0)
    feature_std[feature_std == 0] = 1.0
    classes = np.arange(n_classes)
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])
        beta_true = theta[c_star] - theta[c_runner]

        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        ovr = fit_onevsrest(Z, weights, proba, x)
        beta_ovr = ovr[c_star]["coef"] - ovr[c_runner]["coef"]

        fisher = fit_fisher(Z, weights, hard_labels, classes)
        beta_fisher_hard = fisher["pairwise_direction"](c_star, c_runner)

        fisher_soft = fit_fisher_soft(Z, weights, proba, classes)
        beta_fisher_soft = fisher_soft["pairwise_direction"](c_star, c_runner)

        contrastive = fit_contrastive(Z, weights, proba, c_star, c_runner, x)
        beta_contrastive = contrastive["coef"]

        rows.append(dict(
            n_features=n_features, n_classes=n_classes,
            ovr_spearman=pairwise_coef_spearman(beta_true, beta_ovr),
            fisher_hard_spearman=pairwise_coef_spearman(beta_true, beta_fisher_hard),
            fisher_soft_spearman=pairwise_coef_spearman(beta_true, beta_fisher_soft),
            contrastive_spearman=pairwise_coef_spearman(beta_true, beta_contrastive),
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
    df.to_csv(out_dir / "groundtruth_results.csv", index=False)

    summary = df.groupby(["n_features", "n_classes"]).mean(numeric_only=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-(n_features, n_classes) mean Spearman rho vs true coefficients "
          "(naive, no CI; higher = better ground-truth recovery) ===")
    print(summary)
    print("\n=== overall mean ===")
    print(df.mean(numeric_only=True))

    # --- statistically rigorous comparison ---
    pairs = [
        ("spearman", "contrastive_spearman", "ovr_spearman"),
        ("spearman", "contrastive_spearman", "fisher_hard_spearman"),
        ("spearman", "contrastive_spearman", "fisher_soft_spearman"),
        ("spearman", "ovr_spearman", "fisher_hard_spearman"),
        ("spearman", "ovr_spearman", "fisher_soft_spearman"),
        ("spearman", "fisher_soft_spearman", "fisher_hard_spearman"),
    ]
    stats_df = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats_df.to_csv(out_dir / "groundtruth_stats.csv", index=False)

    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(Holm-Bonferroni corrected across grid cells+pairs; mean_diff = a - b, "
          "positive means a recovers the true ranking BETTER) ===")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
