"""Summarize and plot the two-class standard-LIME baseline experiment.

Input:  results/ovo_vs_ovr_results.csv
Output: results/pairwise_lime_baseline_summary.csv
        results/pairwise_lime_baseline_overall_stats.csv
        results/pairwise_lime_baseline_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from stats_utils import bootstrap_ci, paired_wilcoxon  # noqa: E402


ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"

LABELS = {
    "ovr_union": "OVR和集合\n（平均7.35特徴）",
    "ovr_union_half": "OVR半分和集合\n（平均4.19特徴）",
    "ovr_exact": "OVR\n（ちょうどK特徴）",
    "pairwise_lime": "2クラスを選んで\n通常LIME",
    "contrastive": "対数比＋リッジ回帰\n（logit(q)）",
    "logistic": "局所ロジスティック回帰\n（交差エントロピー）",
}

COLORS = {
    "ovr_union": "#6B7280",
    "ovr_union_half": "#A8ADB5",
    "ovr_exact": "#6B7280",
    "pairwise_lime": "#176B87",
    "contrastive": "#D97706",
    "logistic": "#7C3AED",
}


def main():
    plt.rcParams.update({"font.family": "YuGothic", "axes.unicode_minus": False})
    df = pd.read_csv(RESULTS / "ovo_vs_ovr_results.csv")
    # One independent unit per dataset seed; the 18 grid/K cells and their
    # eight explanation instances are averaged within seed first.
    seed = df.groupby("seed", as_index=False).mean(numeric_only=True)

    rows = []
    for method in LABELS:
        fidelity = seed[f"{method}_fidelity_test"].to_numpy()
        mean, lo, hi = bootstrap_ci(fidelity)
        rows.append(dict(method=method, metric="held_out_sign_fidelity",
                         mean=mean, ci_lo=lo, ci_hi=hi, n_seeds=len(fidelity)))
        brier_col = f"{method}_brier_test"
        if brier_col in seed:
            brier = seed[brier_col].to_numpy()
            mean, lo, hi = bootstrap_ci(brier)
            rows.append(dict(method=method, metric="held_out_weighted_brier",
                             mean=mean, ci_lo=lo, ci_hi=hi, n_seeds=len(brier)))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "pairwise_lime_baseline_summary.csv", index=False)

    comparisons = []
    for metric, pairs in {
        "held_out_sign_fidelity": [
            ("contrastive", "pairwise_lime"),
            ("logistic", "pairwise_lime"),
            ("pairwise_lime", "ovr_exact"),
            ("contrastive", "ovr_exact"),
            ("logistic", "ovr_exact"),
        ],
        "held_out_weighted_brier": [
            ("contrastive", "pairwise_lime"),
            ("logistic", "pairwise_lime"),
            ("logistic", "contrastive"),
        ],
    }.items():
        suffix = "fidelity_test" if metric.endswith("fidelity") else "brier_test"
        for a, b in pairs:
            av = seed[f"{a}_{suffix}"].to_numpy()
            bv = seed[f"{b}_{suffix}"].to_numpy()
            test = paired_wilcoxon(av, bv)
            _, dlo, dhi = bootstrap_ci(av - bv)
            comparisons.append(dict(
                metric=metric, method_a=a, method_b=b,
                mean_diff=test["mean_diff"], diff_ci_lo=dlo, diff_ci_hi=dhi,
                p_value=test["p_value"], effect_size=test["effect_size"],
                n_seeds=test["n"],
            ))
    pd.DataFrame(comparisons).to_csv(
        RESULTS / "pairwise_lime_baseline_overall_stats.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], "held_out_sign_fidelity",
         ["ovr_exact", "pairwise_lime", "contrastive", "logistic"],
         "未使用摂動でのペア符号忠実度", "高いほど良い", (0.75, 0.81)),
        (axes[1], "held_out_weighted_brier",
         ["pairwise_lime", "contrastive", "logistic"],
         "未使用摂動でのペア確率誤差", "低いほど良い（重み付きBrier）", (0.0148, 0.0175)),
    ]
    for ax, metric, methods, title, subtitle, xlim in panels:
        part = summary[summary.metric == metric].set_index("method")
        y = np.arange(len(methods))[::-1]
        means = np.array([part.loc[m, "mean"] for m in methods])
        lo = np.array([part.loc[m, "ci_lo"] for m in methods])
        hi = np.array([part.loc[m, "ci_hi"] for m in methods])
        xerr = np.vstack([means - lo, hi - means])
        for yi, method, mean, err_lo, err_hi in zip(y, methods, means, xerr[0], xerr[1]):
            ax.errorbar(mean, yi, xerr=np.array([[err_lo], [err_hi]]), fmt="o",
                        markersize=9, color=COLORS[method], ecolor=COLORS[method],
                        elinewidth=2.2, capsize=4, markeredgecolor="white",
                        markeredgewidth=0.8, zorder=3)
            digits = 4 if metric.endswith("brier") else 3
            ax.text(mean + (xlim[1] - xlim[0]) * 0.018, yi,
                    f"{mean:.{digits}f}", va="center", ha="left",
                    fontsize=10, color="#111827", fontweight="bold")
        ax.set_yticks(y, [LABELS[m] for m in methods])
        ax.set_xlim(*xlim)
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.9)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#9CA3AF")
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.tick_params(axis="x", colors="#4B5563")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color="#111827", pad=18)
        ax.text(0, 1.015, subtitle + "・拡大表示・95%ブートストラップ信頼区間",
                transform=ax.transAxes, fontsize=9.5, color="#6B7280", va="bottom")

    fig.suptitle("疎なランダムフォレスト実験における上位2クラス説明",
                 x=0.06, ha="left", fontsize=18, fontweight="bold", color="#111827")
    fig.text(0.06, 0.925,
             "特徴数・クラス数9条件 × K 2条件 × 独立データ20シード・学習に使っていない摂動で評価",
             ha="left", fontsize=10.5, color="#4B5563")
    fig.text(0.06, 0.02,
             "4方式とも表示特徴数をちょうどK個に揃え、同じ摂動を使い、特徴選択後に最終サロゲートを再学習。",
             ha="left", fontsize=9.5, color="#4B5563")
    plt.subplots_adjust(left=0.20, right=0.97, top=0.82, bottom=0.15, wspace=0.45)
    fig.savefig(RESULTS / "pairwise_lime_baseline_comparison.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
