"""Plot the harder nonlinear local-ground-truth benchmark in Japanese."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "challenging_groundtruth_results.csv"
OUTPUT = ROOT / "results" / "challenging_groundtruth_comparison.png"
RECALL_OUTPUT = ROOT / "results" / "challenging_groundtruth_recall.png"
FIDELITY_OUTPUT = ROOT / "results" / "challenging_groundtruth_fidelity.png"
SUMMARY = ROOT / "results" / "challenging_groundtruth_summary.csv"

METHODS = {
    "ovr": ("OVR（ちょうどK特徴）", "#6B7280", "o"),
    "two_class_lime": ("2クラス選択＋通常LIME", "#2563EB", "s"),
    "contrastive": ("対数比＋リッジ回帰", "#D97706", "D"),
    "logistic": ("局所ロジスティック回帰", "#059669", "^"),
}


def bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Collapse the six points that share one black box before inference.
    units = df.groupby(["curvature", "n_classes", "seed"], as_index=False).mean(numeric_only=True)
    for curvature in sorted(units.curvature.unique()):
        for n_classes in sorted(units.n_classes.unique()):
            block = units[(units.curvature == curvature) & (units.n_classes == n_classes)]
            for method in METHODS:
                for metric in ["topk_recall", "fidelity"]:
                    mean, lo, hi = bootstrap(
                        block[f"{method}_{metric}"].to_numpy(),
                        seed=20260910 + int(100 * curvature) + n_classes,
                    )
                    rows.append(dict(curvature=curvature, n_classes=n_classes,
                                     method=method, metric=metric,
                                     mean=mean, ci_low=lo, ci_high=hi,
                                     n_seeds=len(block)))
    return pd.DataFrame(rows)


def main() -> None:
    summary = summarize(pd.read_csv(INPUT))
    summary.to_csv(SUMMARY, index=False)
    plt.rcParams.update({
        "font.family": "YuGothic", "font.size": 11,
        "axes.unicode_minus": False, "axes.titlesize": 14,
        "axes.labelsize": 11, "legend.fontsize": 10,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.3), sharex=True,
                             constrained_layout=True)
    curvatures = [0.35, 0.80]
    metric_info = [
        ("topk_recall", "真の局所Top-5特徴の再現率", (0.70, 0.98)),
        ("fidelity", "未使用摂動でのA/B符号忠実度", (0.84, 0.93)),
    ]
    x = np.arange(3)
    offsets = np.linspace(-0.18, 0.18, len(METHODS))
    for row, curvature in enumerate(curvatures):
        for col, (metric, title, ylim) in enumerate(metric_info):
            ax = axes[row, col]
            for offset, (method, (label, color, marker)) in zip(offsets, METHODS.items()):
                part = (summary[(summary.curvature == curvature) &
                                (summary.metric == metric) &
                                (summary.method == method)]
                        .set_index("n_classes").loc[[3, 5, 8]])
                mean = part["mean"].to_numpy()
                err = np.vstack([mean - part["ci_low"].to_numpy(),
                                 part["ci_high"].to_numpy() - mean])
                ax.errorbar(x + offset, mean, yerr=err, label=label, color=color,
                            marker=marker, markersize=6, linewidth=1.7,
                            capsize=3, capthick=1.2)
            ax.set_ylim(*ylim)
            ax.set_xticks(x, ["3", "5", "8"])
            ax.set_xlabel("クラス数")
            ax.set_ylabel("高いほど良い")
            ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, loc="left", weight="bold")
            ax.text(0.98, 0.05, f"曲率={curvature:.2f}", transform=ax.transAxes,
                    ha="right", va="bottom", color="#475467", weight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("非線形・相関・ノイズを含む局所グラウンドトゥルース実験",
                 x=0.02, ha="left", fontsize=18, weight="bold")
    fig.text(0.98, 0.965,
             "30次元・K=5・各条件20独立BB × 6説明点・平均と95%ブートストラップ信頼区間",
             ha="right", va="top", fontsize=10, color="#475467")
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")

    # Slide-friendly single-metric figures.  Keeping weak and strong curvature
    # side by side makes the robustness check visible without shrinking text.
    for metric, title, ylim, out_path in [
        ("topk_recall", "真の局所Top-5特徴の再現率", (0.70, 0.98), RECALL_OUTPUT),
        ("fidelity", "未使用摂動でのA/B符号忠実度", (0.84, 0.93), FIDELITY_OUTPUT),
    ]:
        f, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True,
                              constrained_layout=True)
        for ax, curvature in zip(axs, curvatures):
            for offset, (method, (label, color, marker)) in zip(offsets, METHODS.items()):
                part = (summary[(summary.curvature == curvature) &
                                (summary.metric == metric) &
                                (summary.method == method)]
                        .set_index("n_classes").loc[[3, 5, 8]])
                mean = part["mean"].to_numpy()
                err = np.vstack([mean - part["ci_low"].to_numpy(),
                                 part["ci_high"].to_numpy() - mean])
                ax.errorbar(x + offset, mean, yerr=err, label=label, color=color,
                            marker=marker, markersize=7, linewidth=1.9,
                            capsize=3, capthick=1.3)
            ax.set_ylim(*ylim)
            ax.set_xticks(x, ["3", "5", "8"])
            ax.set_xlabel("クラス数")
            ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(f"曲率={curvature:.2f}", loc="left", weight="bold")
        axs[0].set_ylabel("高いほど良い")
        h, lab = axs[0].get_legend_handles_labels()
        f.legend(h, lab, loc="outside lower center", ncol=4, frameon=False)
        f.suptitle(title, x=0.02, ha="left", fontsize=18, weight="bold")
        f.text(0.98, 0.965, "30次元・K=5・各条件20独立BB × 6説明点・95%信頼区間",
               ha="right", va="top", fontsize=10, color="#475467")
        f.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(f)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(OUTPUT)


if __name__ == "__main__":
    main()
