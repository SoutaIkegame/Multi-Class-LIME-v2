"""Does Contrastive LIME's fidelity advantage over one-vs-rest concentrate
in the extreme-probability regime (one of the two competing classes' local
probability near 0), rather than showing up uniformly?

Rationale: log((p_c1+eps)/(p_c2+eps)) and p_c1-p_c2 have the same sign
everywhere (so the earlier pairwise-sign fidelity check couldn't tell them
apart in the moderate regime), but they behave very differently when one
probability is tiny: the log-odds diverges while the raw difference stays
small and near-constant, potentially harder for a bounded linear model to
resolve. This splits the SAME already-fit local models' predictions by
whether the perturbed point z falls in an "extreme" (min(p_c*,p_c') below
a threshold) or "moderate" region, and compares sign-fidelity separately.

Usage: python3 src/run_extreme_regime_experiment.py
Output: results/extreme_regime_results.csv (+ printed summary)
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
from surrogates import fit_onevsrest, fit_fisher, fit_contrastive  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
EXTREME_THRESHOLD = 0.15
SEED = 0
# See src/stats_utils.py docstring: each grid cell is re-run over this many
# independent dataset draws so method comparisons can be tested for
# significance instead of read off a single draw. run_one_cell() already
# averages over the N_INSTANCES contested instances internally, so each
# call below IS one seed-level replicate.
N_DATASET_SEEDS = 20


def weighted_sign_accuracy(pred_sign, true_sign, weights, mask):
    if pred_sign is None or mask.sum() == 0:
        return float("nan")
    return float(np.average(pred_sign[mask] == true_sign[mask], weights=weights[mask]))


def run_one_cell(n_features: int, n_classes: int, rng: np.random.Generator) -> dict:
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

    frac_extreme = []
    extreme = {"ovr": [], "fisher": [], "contrastive": []}
    moderate = {"ovr": [], "fisher": [], "contrastive": []}

    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c_star, c_runner = int(order[0]), int(order[1])

        Z, weights = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        hard_labels = proba.argmax(axis=1)

        min_p = np.minimum(proba[:, c_star], proba[:, c_runner])
        extreme_mask = min_p < EXTREME_THRESHOLD
        moderate_mask = ~extreme_mask
        frac_extreme.append(np.average(extreme_mask, weights=weights))
        true_sign = np.sign(proba[:, c_star] - proba[:, c_runner])

        ovr = fit_onevsrest(Z, weights, proba, x)
        ovr_scores = Z @ (ovr[c_star]["coef"] - ovr[c_runner]["coef"]) + \
            (ovr[c_star]["intercept"] - ovr[c_runner]["intercept"])
        ovr_sign = np.sign(ovr_scores)

        fisher = fit_fisher(Z, weights, hard_labels, classes)
        v = fisher["pairwise_direction"](c_star, c_runner)
        fisher_sign = None
        if v is not None and c_star in fisher["mu"] and c_runner in fisher["mu"]:
            midpoint = 0.5 * (fisher["mu"][c_star] + fisher["mu"][c_runner])
            fisher_sign = np.sign((Z - midpoint[None, :]) @ v)

        contrastive = fit_contrastive(Z, weights, proba, c_star, c_runner, x)
        contrastive_scores = Z @ contrastive["coef"] + contrastive["intercept"]
        contrastive_sign = np.sign(contrastive_scores)

        for name, sign in [("ovr", ovr_sign), ("fisher", fisher_sign), ("contrastive", contrastive_sign)]:
            extreme[name].append(weighted_sign_accuracy(sign, true_sign, weights, extreme_mask))
            moderate[name].append(weighted_sign_accuracy(sign, true_sign, weights, moderate_mask))

    return dict(
        n_features=n_features, n_classes=n_classes,
        frac_extreme=float(np.nanmean(frac_extreme)),
        extreme_ovr=float(np.nanmean(extreme["ovr"])),
        extreme_fisher=float(np.nanmean(extreme["fisher"])),
        extreme_contrastive=float(np.nanmean(extreme["contrastive"])),
        moderate_ovr=float(np.nanmean(moderate["ovr"])),
        moderate_fisher=float(np.nanmean(moderate["fisher"])),
        moderate_contrastive=float(np.nanmean(moderate["contrastive"])),
    )


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    n_cells = len(N_FEATURES_GRID) * len(N_CLASSES_GRID)
    cell_i = 0
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            cell_i += 1
            print(f"[{time.time()-t0:6.1f}s] cell {cell_i}/{n_cells}: n_features={n_features}, "
                  f"n_classes={n_classes}, {N_DATASET_SEEDS} dataset seeds ...")
            for seed in range(N_DATASET_SEEDS):
                row = run_one_cell(n_features, n_classes, rng)
                row["seed"] = seed
                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "extreme_regime_results.csv", index=False)

    pd.set_option("display.width", 200)
    summary = df.groupby(["n_features", "n_classes"]).mean(numeric_only=True)
    print("\n=== per grid cell mean over seeds (naive, no CI) ===")
    print(summary)
    print("\n=== overall mean ===")
    print(df.mean(numeric_only=True))

    # --- statistically rigorous comparison ---
    pairs = [
        ("extreme_fidelity", "extreme_contrastive", "extreme_ovr"),
        ("extreme_fidelity", "extreme_fisher", "extreme_ovr"),
        ("moderate_fidelity", "moderate_contrastive", "moderate_ovr"),
        ("moderate_fidelity", "moderate_fisher", "moderate_ovr"),
    ]
    stats_df = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats_df.to_csv(out_dir / "extreme_regime_stats.csv", index=False)

    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(Holm-Bonferroni corrected across grid cells+pairs) ===")
    print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
