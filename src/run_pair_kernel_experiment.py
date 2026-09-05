"""Proposal B: pair-aware ("contest-weighted") kernel for Contrastive LIME.

LIME's kernel pi_i weights perturbations by distance to x only. For a
pairwise explanation of c* vs c', perturbations where one of the two
classes is already hopeless (q = p_c*/(p_c*+p_c') near 0 or 1) carry
little information about the c*-vs-c' boundary, and the log-odds target
diverges there, which is exactly where run_extreme_regime_experiment.py
found the surrogates struggling. Proposal: multiply the kernel by the
Bernoulli variance of the pair contest,

    pi'_i = pi_i * (4 q_i (1 - q_i) + floor),

so the fit concentrates on the region where the two classes actually
compete. This is pair-conditional (differs from CLIMAX's class-balanced
perturbation, which is one-vs-rest and resamples rather than reweights).

Evaluation: same RF black box and extreme/moderate split as
run_extreme_regime_experiment.py. Both variants are scored with the SAME
standard kernel pi so the comparison is on identical ground; fidelity is
reported on the whole neighborhood, the extreme region, and the moderate
region, plus normalized direction variance under resampling.

CAVEAT (found 2026-09-05): fidelity/extreme/moderate below are measured on
the SAME Z used to fit each variant -- in-sample fit quality, not evidence
of generalization to an unseen neighborhood. direction_variance is fine (it
already comes from independent resampling repeats). The properly held-out
re-measurement of this exact comparison (plus proposal C and the B+C
combination) lives in run_combined_bc_experiment.py's *_test columns --
treat that script's numbers as authoritative for B's fidelity claims, not
this one's.

Usage: python3 src/run_pair_kernel_experiment.py
Output: results/pair_kernel_{results,stats}.csv
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
from surrogates import fit_contrastive  # noqa: E402
from metrics import total_variance_normalized  # noqa: E402
from run_experiment import pick_contested_instances  # noqa: E402
from stats_utils import compare_methods  # noqa: E402

N_FEATURES_GRID = [8, 14, 20]
N_CLASSES_GRID = [3, 4, 5]
N_INSTANCES = 8
N_PERTURB_SAMPLES = 300
N_STABILITY_REPEATS = 8
EXTREME_THRESHOLD = 0.15
FLOOR = 0.05
SEED = 0
N_DATASET_SEEDS = 20


def contest_weights(weights, proba, c1, c2, floor=FLOOR):
    q = proba[:, c1] / (proba[:, c1] + proba[:, c2] + 1e-12)
    return weights * (4.0 * q * (1.0 - q) + floor)


def _sign_acc(coef, b, Z, proba, c1, c2, w, mask=None):
    pred = np.sign(Z @ coef + b)
    true = np.sign(proba[:, c1] - proba[:, c2])
    if mask is None:
        mask = np.ones(len(w), dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(pred[mask] == true[mask], weights=w[mask]))


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

    rows = []
    for x in instances:
        x_proba = clf.predict_proba(x[None, :])[0]
        order = np.argsort(x_proba)[::-1]
        c1, c2 = int(order[0]), int(order[1])

        Z, w = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
        proba = clf.predict_proba(Z)
        extreme = np.minimum(proba[:, c1], proba[:, c2]) < EXTREME_THRESHOLD
        fits = {
            "standard": fit_contrastive(Z, w, proba, c1, c2, x),
            "pairkernel": fit_contrastive(Z, contest_weights(w, proba, c1, c2), proba, c1, c2, x),
        }
        row = dict(n_features=n_features, n_classes=n_classes,
                   frac_extreme=float(np.average(extreme, weights=w)))
        for m, f in fits.items():
            row[f"{m}_fidelity"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w)
            row[f"{m}_extreme"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, extreme)
            row[f"{m}_moderate"] = _sign_acc(f["coef"], f["intercept"], Z, proba, c1, c2, w, ~extreme)

        hist = {m: [fits[m]["coef"]] for m in fits}
        for _ in range(N_STABILITY_REPEATS - 1):
            Zk, wk = sample_perturbations(x, feature_std, N_PERTURB_SAMPLES, rng)
            pk = clf.predict_proba(Zk)
            hist["standard"].append(fit_contrastive(Zk, wk, pk, c1, c2, x)["coef"])
            hist["pairkernel"].append(fit_contrastive(Zk, contest_weights(wk, pk, c1, c2), pk, c1, c2, x)["coef"])
        for m in fits:
            row[f"{m}_direction_variance"] = total_variance_normalized(hist[m])
        rows.append(row)
    return rows


def main():
    rng = np.random.default_rng(SEED)
    all_rows = []
    t0 = time.time()
    for n_features in N_FEATURES_GRID:
        for n_classes in N_CLASSES_GRID:
            print(f"[{time.time()-t0:6.1f}s] n_features={n_features}, n_classes={n_classes}", flush=True)
            for seed in range(N_DATASET_SEEDS):
                rows = run_one_cell(n_features, n_classes, rng)
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    out = Path(__file__).parent.parent / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "pair_kernel_results.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    cols = [c for c in df.columns if c not in ("n_features", "n_classes", "seed")]
    print("\n=== naive means ===")
    print(df.groupby(["n_features", "n_classes"])[cols].mean().round(4))
    print("\n=== overall ===")
    print(df[cols].mean().round(4))

    pairs = [(m, f"pairkernel_{m}", f"standard_{m}")
             for m in ["fidelity", "extreme", "moderate", "direction_variance"]]
    stats = compare_methods(df, ["n_features", "n_classes"], pairs)
    stats.to_csv(out / "pair_kernel_stats.csv", index=False)
    print("\n=== paired tests (20 seeds, Holm within metric); mean_diff = pairkernel - standard ===")
    print(stats[["n_features", "n_classes", "metric", "mean_a", "mean_b", "mean_diff",
                 "p_value", "effect_size", "p_value_holm_reject"]].to_string(index=False))


if __name__ == "__main__":
    main()
