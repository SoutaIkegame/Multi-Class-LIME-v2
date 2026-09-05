"""Statistical-rigor helpers shared by the experiment drivers.

Why this module exists
-----------------------
Every ``run_*_experiment.py`` script used to draw exactly ONE synthetic
dataset + black-box classifier per (n_features, n_classes) grid cell, then
average a metric over 8 "contested" instances drawn from that single
dataset. Those 8 instances are not independent replicates of the
experiment -- they all come from the same classifier fit on the same data
draw, so their spread only reflects within-dataset instance-to-instance
variation, not run-to-run (dataset draw + classifier fit) variation. Two
consequences followed: (1) no error bars could honestly be reported on the
grid-cell means, and (2) comparing methods with a plain mean gave no way to
tell a real effect from noise in that one draw.

The fix used throughout this codebase now: each grid cell is repeated over
``N_DATASET_SEEDS`` independent dataset draws (see ``DATASET_SEEDS`` in each
script). Within one seed, the per-instance rows are first averaged down to
a single per-seed cell mean (this is what ``seed_level_means`` does) --
that collapses the non-independent instances into one number per
independent replicate, so the N_DATASET_SEEDS seed-level means can be
treated as i.i.d. samples. All confidence intervals and significance tests
below operate on those seed-level means, never on the raw per-instance
rows, to avoid pseudo-replication (inflated apparent sample size / falsely
narrow confidence intervals from treating correlated instances as
independent).

Since the same dataset seed (and, for paired metrics, the same perturbation
draw) is shared across methods being compared, method-vs-method comparisons
use paired tests (Wilcoxon signed-rank), not two-sample tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def seed_level_means(df: pd.DataFrame, group_cols: list[str], seed_col: str = "seed") -> pd.DataFrame:
    """Collapse per-instance rows to one mean row per (seed, *group_cols).

    This is the pseudo-replication fix: instances within one dataset seed
    are correlated (same classifier, same data draw), so they must be
    averaged into a single number per seed before that seed is treated as
    an independent sample.
    """
    return df.groupby([seed_col] + group_cols, as_index=False).mean(numeric_only=True)


def bootstrap_ci(values, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap mean + 95% CI over independent samples (e.g.
    seed-level cell means, NOT raw per-instance rows)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def paired_wilcoxon(a, b) -> dict:
    """Paired Wilcoxon signed-rank test between two methods' seed-level
    means. ``a[i]`` and ``b[i]`` must come from the SAME seed (and, ideally,
    the same underlying perturbation draw) so the pairing is valid.

    Returns n pairs used, the test statistic, the p-value, a matched-pairs
    rank-biserial effect size in [-1, 1] (sign = direction, magnitude =
    how one-sided the paired differences are -- NOT the same thing as
    "practically large"; report alongside mean_diff), and the mean paired
    difference (a - b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    n = len(diff)
    if n < 2:
        return dict(n=n, statistic=float("nan"), p_value=float("nan"),
                    effect_size=float("nan"),
                    mean_diff=float(diff[0]) if n == 1 else float("nan"))
    if np.allclose(diff, 0.0):
        return dict(n=n, statistic=float("nan"), p_value=1.0, effect_size=0.0,
                    mean_diff=0.0)
    try:
        stat, p = sps.wilcoxon(diff, zero_method="wilcox")
    except ValueError:
        # all remaining differences are exactly zero after dropping ties
        return dict(n=n, statistic=float("nan"), p_value=1.0, effect_size=0.0,
                    mean_diff=float(np.mean(diff)))
    nz = diff[diff != 0]
    ranks = sps.rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    total = r_plus + r_minus
    effect = (r_plus - r_minus) / total if total > 0 else 0.0
    return dict(n=n, statistic=float(stat), p_value=float(p), effect_size=float(effect),
                mean_diff=float(np.mean(diff)))


def holm_bonferroni(pvalues) -> np.ndarray:
    """Holm-Bonferroni step-down correction for a family of tests.

    Returns a boolean array (same order as input) marking which hypotheses
    are rejected at family-wise alpha=0.05. NaN p-values are never
    rejected.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(np.where(np.isnan(pvalues), np.inf, pvalues))
    reject = np.zeros(n, dtype=bool)
    for rank, idx in enumerate(order):
        p = pvalues[idx]
        if np.isnan(p):
            continue
        threshold = 0.05 / (n - rank)
        if p <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def compare_methods(df: pd.DataFrame, group_cols: list[str], metric_pairs: list[tuple[str, str, str]],
                     seed_col: str = "seed") -> pd.DataFrame:
    """Run the full pipeline (seed-level aggregation -> per-cell bootstrap
    CI for each method -> paired Wilcoxon between methods -> Holm-Bonferroni
    correction across grid cells) for a set of named method-vs-method
    metric comparisons.

    Parameters
    ----------
    df : raw per-instance rows, must contain ``seed_col`` and all
        ``group_cols`` plus every column name referenced in ``metric_pairs``.
    group_cols : grid columns identifying a cell, e.g. ["n_features", "n_classes"]
        or ["n_features", "n_classes", "K"].
    metric_pairs : list of (metric_name, col_a, col_b); col_a/col_b are the
        two methods' columns for that metric (e.g. "ovr_feature_overlap",
        "fisher_feature_overlap"). The reported difference is a - b.

    Returns
    -------
    One row per (metric_name, grid cell): seed-level means + CI for each
    method, the paired difference, raw and Holm-corrected p-values, and the
    effect size.
    """
    seed_means = seed_level_means(df, group_cols, seed_col)
    rows = []
    for metric_name, col_a, col_b in metric_pairs:
        for cell_vals, cell_df in seed_means.groupby(group_cols):
            cell_vals = cell_vals if isinstance(cell_vals, tuple) else (cell_vals,)
            a = cell_df[col_a].to_numpy()
            b = cell_df[col_b].to_numpy()
            mean_a, lo_a, hi_a = bootstrap_ci(a)
            mean_b, lo_b, hi_b = bootstrap_ci(b)
            test = paired_wilcoxon(a, b)
            row = dict(zip(group_cols, cell_vals))
            row.update(dict(
                metric=metric_name, method_a=col_a, method_b=col_b,
                n_seeds=test["n"],
                mean_a=mean_a, ci_lo_a=lo_a, ci_hi_a=hi_a,
                mean_b=mean_b, ci_lo_b=lo_b, ci_hi_b=hi_b,
                mean_diff=test["mean_diff"], p_value=test["p_value"],
                effect_size=test["effect_size"],
            ))
            rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["p_value_holm_reject"] = False
        for metric_name in result["metric"].unique():
            mask = result["metric"] == metric_name
            result.loc[mask, "p_value_holm_reject"] = holm_bonferroni(result.loc[mask, "p_value"].to_numpy())
    return result
