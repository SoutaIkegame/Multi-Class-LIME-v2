"""Central research question (see docs/OVO_LIME_METHODS.md, 2026-09-06):
does directly approximating a class PAIR (OVO) explain the black box's
preference between those two classes more faithfully than an OVR-style
explanation, AT THE SAME DISPLAYED FEATURE COUNT K?

Four methods:

  ovr_union       Each of c1, c2 gets its own independent top-K Lasso
                  one-vs-rest explanation (fit_onevsrest_lasso); the pair
                  explanation is the coefficient DIFFERENCE, whose support
                  is the UNION of the two classes' selected features (size
                  K to 2K, not exactly K -- an OVR-based pairwise
                  explanation cannot be forced below 2K without dropping
                  one class's own features, so the union size is reported
                  honestly as that method's actual complexity rather than
                  pretending it's K). This gives OVR a complexity
                  ADVANTAGE over the two OVO methods below (confirmed
                  empirically: ~1.3-1.4x K on average).
  ovr_union_half  CONTROL for that advantage (added 2026-09-06): same
                  construction, but each class only gets ceil(K/2) Lasso
                  features, so the union lands close to K instead of well
                  above it. If ovr_union's parity with Contrastive was
                  just its complexity handicap, this version -- fit at
                  matched complexity -- should show Contrastive pulling
                  ahead.
  contrastive_K   fit_contrastive_lasso: log-ratio Ridge restricted to
                  exactly K features, selected jointly for the pair.
  logistic_K      fit_ovo_logistic_lasso: soft-label logistic restricted
                  to exactly K features, selected jointly for the pair.

Fidelity is pairwise-sign agreement, measured on an INDEPENDENT held-out
perturbation sample (never used for fitting) -- see run_combined_bc_
experiment.py's docstring for why this matters. Reported at multiple K to
trace out a fidelity-vs-complexity curve, which is the form the central
question actually takes ("fewer features, still faithful").

This tests hypothesis 1 from the 2026-09-06 review only (does OVO
reproduce the pairwise decision more faithfully at equal display cost).
Hypotheses 2 (does OVO's selected support avoid class-common features and
favor pair-specific ones) and 3 (does the explanation change appropriately
when the competitor class changes) need a synthetic generator with an
explicit common/pair-specific/irrelevant feature split and are NOT
implemented here -- left as follow-up work.

Usage: python3 src/run_ovo_vs_ovr_experiment.py
Output: results/ovo_vs_ovr_{results,stats}.csv
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
from surrogates import fit_onevsrest_lasso, fit_contrastive_lasso, fit_ovo_logistic_lasso  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
K_FRACS = [0.25, 0.5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
SEED = 0
N_DATASET_SEEDS = 20


def _sign_acc(coef, b, Z, proba, c1, c2, w):
    pred = np.sign(Z @ coef + b)
    true = np.sign(proba[:, c1] - proba[:, c2])
    return float(np.average(pred == true, weights=w))


def run_one_cell(n_features, n_classes, rng):
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
    instances = pick_contested_instances(clf, X_test, N_INSTANCES)
    k_values = sorted({max(1, round(f * n_features)) for f in K_FRACS})

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])

        Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)

        # independent held-out neighborhood, never touched by fitting
        Z_test, w_test = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba_test = clf.predict_proba(Z_test)

        for K in k_values:
            ovr_lasso = fit_onevsrest_lasso(Z, w, proba, x, K)
            ovr1, ovr2 = ovr_lasso[c1], ovr_lasso[c2]
            ovr_coef = ovr1["coef"] - ovr2["coef"]
            ovr_intercept = ovr1["intercept"] - ovr2["intercept"]
            ovr_union_size = len(ovr1["selected"] | ovr2["selected"])

            K_half = max(1, -(-K // 2))  # ceil(K/2)
            ovr_lasso_half = fit_onevsrest_lasso(Z, w, proba, x, K_half)
            ovrh1, ovrh2 = ovr_lasso_half[c1], ovr_lasso_half[c2]
            ovrh_coef = ovrh1["coef"] - ovrh2["coef"]
            ovrh_intercept = ovrh1["intercept"] - ovrh2["intercept"]
            ovrh_union_size = len(ovrh1["selected"] | ovrh2["selected"])

            con = fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)
            log = fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)

            rows.append(dict(
                n_features=n_features, n_classes=n_classes, K=K,
                ovr_union_fidelity_test=_sign_acc(ovr_coef, ovr_intercept, Z_test, proba_test, c1, c2, w_test),
                ovr_union_complexity=ovr_union_size,
                ovr_union_half_fidelity_test=_sign_acc(ovrh_coef, ovrh_intercept, Z_test, proba_test, c1, c2, w_test),
                ovr_union_half_complexity=ovrh_union_size,
                contrastive_fidelity_test=_sign_acc(con["coef"], con["intercept"], Z_test, proba_test, c1, c2, w_test),
                logistic_fidelity_test=_sign_acc(log["coef"], log["intercept"], Z_test, proba_test, c1, c2, w_test),
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
                  f"n_classes={n_classes}, {N_DATASET_SEEDS} dataset seeds ...", flush=True)
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "ovo_vs_ovr_results.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    cols = [c for c in df.columns if c not in ("n_features", "n_classes", "K", "seed")]
    print("\n=== naive means per (n_features, n_classes, K) ===")
    print(df.groupby(["n_features", "n_classes", "K"])[cols].mean().round(4))
    print("\n=== overall ===")
    print(df[cols].mean().round(4))

    pairs = [
        ("fidelity_test", "contrastive_fidelity_test", "ovr_union_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "ovr_union_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "contrastive_fidelity_test"),
        ("fidelity_test", "contrastive_fidelity_test", "ovr_union_half_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "ovr_union_half_fidelity_test"),
        ("fidelity_test", "ovr_union_fidelity_test", "ovr_union_half_fidelity_test"),
    ]
    stats = compare_methods(df, ["n_features", "n_classes", "K"], pairs)
    stats.to_csv(out / "ovo_vs_ovr_stats.csv", index=False)
    print(f"\n=== paired tests across {N_DATASET_SEEDS} independent dataset seeds "
          "(held-out fidelity, Holm-Bonferroni corrected); mean_diff = a - b ===")
    print(stats[["n_features", "n_classes", "K", "metric", "method_a", "method_b",
                 "mean_a", "mean_b", "mean_diff", "p_value", "effect_size",
                 "p_value_holm_reject"]].to_string(index=False))


if __name__ == "__main__":
    main()
