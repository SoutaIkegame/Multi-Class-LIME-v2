"""Central research question (see docs/OVO_LIME_METHODS.md, 2026-09-06):
does directly approximating a class PAIR (OVO) explain the black box's
preference between those two classes more faithfully than an OVR-style
explanation, AT THE SAME DISPLAYED FEATURE COUNT K?

Six methods:

  ovr_union       Each of c1, c2 gets its own independent top-K Lasso
                  feature selection followed by a Ridge refit; the pair
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
  ovr_exact_K     Start from the two independently selected OVR supports,
                  rank their union by the magnitude of the fitted coefficient
                  difference, retain exactly K distinct displayed features,
                  and refit both class-probability surrogates on that support.
                  This is the exact-complexity OVR baseline for the headline
                  comparison; the union and half-union variants remain as
                  sensitivity bounds for the unavoidable merge choice.
  pairwise_lime_K Select c1 and c2, renormalize their black-box probabilities
                  to q=p_c1/(p_c1+p_c2), then apply an ordinary weighted
                  linear LIME surrogate to q with exactly K displayed
                  features. This separates the benefit of choosing a pair
                  from the benefit of the logit link / logistic loss.
  contrastive_K   Lasso-select exactly K features on the log-ratio target,
                  then refit weighted Ridge on those features.
  logistic_K      L1-logistic-select exactly K features, then refit weighted
                  soft-label logistic regression on those features.

Every sparse method follows the same select-then-refit structure. This is
important: LIME uses its sparse model/path only to select features and then
fits the final local surrogate on that support. Using coefficients from the
strongly regularized selector directly would confound target/link effects
with shrinkage and was the cause of an unfair intermediate implementation of
this experiment.

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
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from perturbation import sample_perturbations  # noqa: E402
from surrogates import (  # noqa: E402
    fit_onevsrest_lasso,
    fit_pairwise_lime_lasso,
    fit_contrastive_lasso,
    fit_ovo_logistic_lasso,
)
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


def _pair_q(proba, c1, c2, eps=1e-6):
    return (proba[:, c1] + eps) / (proba[:, c1] + proba[:, c2] + 2 * eps)


def _weighted_brier(pred_q, true_q, w):
    return float(np.average((np.clip(pred_q, 0.0, 1.0) - true_q) ** 2, weights=w))


def _ridge_refit(Z, weights, y, selected, x, alpha=1.0):
    idx = np.array(sorted(selected), dtype=int)
    model = Ridge(alpha=alpha)
    model.fit(Z[:, idx], y, sample_weight=weights)
    coef = np.zeros(Z.shape[1], dtype=float)
    coef[idx] = model.coef_
    intercept = float(model.intercept_)
    return {"coef": coef, "intercept": intercept,
            "local_pred": float(intercept + coef @ x), "selected": frozenset(idx.tolist())}


def _soft_logistic_refit(Z, weights, q, selected, x, C=1.0):
    idx = np.array(sorted(selected), dtype=int)
    Zs = Z[:, idx]
    Z2 = np.vstack([Zs, Zs])
    y2 = np.concatenate([np.ones(len(Z)), np.zeros(len(Z))])
    w2 = np.concatenate([weights * q, weights * (1.0 - q)])
    model = LogisticRegression(C=C, max_iter=2000, random_state=0)
    model.fit(Z2, y2, sample_weight=w2)
    coef = np.zeros(Z.shape[1], dtype=float)
    coef[idx] = model.coef_[0]
    intercept = float(model.intercept_[0])
    return {"coef": coef, "intercept": intercept,
            "local_pred": float(intercept + coef @ x), "selected": frozenset(idx.tolist())}


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
            ovr1 = _ridge_refit(Z, w, proba[:, c1], ovr_lasso[c1]["selected"], x)
            ovr2 = _ridge_refit(Z, w, proba[:, c2], ovr_lasso[c2]["selected"], x)
            ovr_coef = ovr1["coef"] - ovr2["coef"]
            ovr_intercept = ovr1["intercept"] - ovr2["intercept"]
            ovr_union_size = len(ovr1["selected"] | ovr2["selected"])

            # Exact-K OVR display: merge the independently selected class-wise
            # supports, keep the K largest fitted pair-difference terms, and
            # refit both original OVR targets on that common displayed set.
            # This preserves the OVR targets while matching the final number
            # of distinct features shown by all three pair-target methods.
            ovr_candidates = np.array(sorted(ovr1["selected"] | ovr2["selected"]), dtype=int)
            ovr_candidate_scores = np.abs(ovr_coef[ovr_candidates])
            ovr_exact_idx = ovr_candidates[np.argsort(-ovr_candidate_scores)[:K]]
            ovr_exact_selected = frozenset(ovr_exact_idx.tolist())
            ovre1 = _ridge_refit(Z, w, proba[:, c1], ovr_exact_selected, x)
            ovre2 = _ridge_refit(Z, w, proba[:, c2], ovr_exact_selected, x)
            ovre_coef = ovre1["coef"] - ovre2["coef"]
            ovre_intercept = ovre1["intercept"] - ovre2["intercept"]

            K_half = max(1, -(-K // 2))  # ceil(K/2)
            ovr_lasso_half = fit_onevsrest_lasso(Z, w, proba, x, K_half)
            ovrh1 = _ridge_refit(Z, w, proba[:, c1], ovr_lasso_half[c1]["selected"], x)
            ovrh2 = _ridge_refit(Z, w, proba[:, c2], ovr_lasso_half[c2]["selected"], x)
            ovrh_coef = ovrh1["coef"] - ovrh2["coef"]
            ovrh_intercept = ovrh1["intercept"] - ovrh2["intercept"]
            ovrh_union_size = len(ovrh1["selected"] | ovrh2["selected"])

            pair_lime = fit_pairwise_lime_lasso(Z, w, proba, c1, c2, x, K)
            con_selector = fit_contrastive_lasso(Z, w, proba, c1, c2, x, K)
            log_selector = fit_ovo_logistic_lasso(Z, w, proba, c1, c2, x, K)
            eps = 1e-6
            log_ratio = np.log((proba[:, c1] + eps) / (proba[:, c2] + eps))
            q_train = _pair_q(proba, c1, c2, eps)
            con = _ridge_refit(Z, w, log_ratio, con_selector["selected"], x)
            log = _soft_logistic_refit(Z, w, q_train, log_selector["selected"], x)
            true_q_test = _pair_q(proba_test, c1, c2)
            pair_lime_linear_test = Z_test @ pair_lime["coef"] + pair_lime["intercept"]
            con_q_test = expit(Z_test @ con["coef"] + con["intercept"])
            log_q_test = expit(Z_test @ log["coef"] + log["intercept"])

            rows.append(dict(
                n_features=n_features, n_classes=n_classes, K=K,
                ovr_union_fidelity_test=_sign_acc(ovr_coef, ovr_intercept, Z_test, proba_test, c1, c2, w_test),
                ovr_union_complexity=ovr_union_size,
                ovr_exact_fidelity_test=_sign_acc(ovre_coef, ovre_intercept, Z_test, proba_test, c1, c2, w_test),
                ovr_exact_complexity=len(ovr_exact_selected),
                ovr_union_half_fidelity_test=_sign_acc(ovrh_coef, ovrh_intercept, Z_test, proba_test, c1, c2, w_test),
                ovr_union_half_complexity=ovrh_union_size,
                pairwise_lime_fidelity_test=_sign_acc(
                    pair_lime["coef"], pair_lime["intercept"] - 0.5,
                    Z_test, proba_test, c1, c2, w_test,
                ),
                pairwise_lime_complexity=len(pair_lime["selected"]),
                pairwise_lime_brier_test=_weighted_brier(pair_lime_linear_test, true_q_test, w_test),
                contrastive_fidelity_test=_sign_acc(con["coef"], con["intercept"], Z_test, proba_test, c1, c2, w_test),
                contrastive_complexity=len(con["selected"]),
                contrastive_brier_test=_weighted_brier(con_q_test, true_q_test, w_test),
                logistic_fidelity_test=_sign_acc(log["coef"], log["intercept"], Z_test, proba_test, c1, c2, w_test),
                logistic_complexity=len(log["selected"]),
                logistic_brier_test=_weighted_brier(log_q_test, true_q_test, w_test),
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
        ("fidelity_test", "contrastive_fidelity_test", "pairwise_lime_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "pairwise_lime_fidelity_test"),
        ("fidelity_test", "pairwise_lime_fidelity_test", "ovr_union_fidelity_test"),
        ("fidelity_test", "pairwise_lime_fidelity_test", "ovr_exact_fidelity_test"),
        ("fidelity_test", "pairwise_lime_fidelity_test", "ovr_union_half_fidelity_test"),
        ("brier_test", "contrastive_brier_test", "pairwise_lime_brier_test"),
        ("brier_test", "logistic_brier_test", "pairwise_lime_brier_test"),
        ("brier_test", "logistic_brier_test", "contrastive_brier_test"),
        ("fidelity_test", "contrastive_fidelity_test", "ovr_union_fidelity_test"),
        ("fidelity_test", "contrastive_fidelity_test", "ovr_exact_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "ovr_union_fidelity_test"),
        ("fidelity_test", "logistic_fidelity_test", "ovr_exact_fidelity_test"),
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
