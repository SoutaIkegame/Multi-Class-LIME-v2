"""Plot select-then-refit runtime by dimension and class count."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "timing_results.csv"
OUTPUT = RESULTS / "timing_scaling.png"
SUMMARY = RESULTS / "timing_plot_summary.csv"

METHODS = {
    "ovr_exact_ms": ("OVR（全クラス＋ちょうどK特徴）", "#4B5563", "o"),
    "two_class_lime_ms": ("2クラス選択＋通常LIME", "#2563EB", "s"),
    "contrastive_ms": ("対数比＋リッジ回帰", "#D97706", "D"),
    "logistic_ms": ("局所ロジスティック回帰", "#059669", "^"),
}


def bootstrap(values, seed):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (10_000, len(values)), replace=True).mean(axis=1)
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def summarize(df):
    rows = []
    for sweep, xcol in [("dimension", "n_features"), ("classes", "n_classes")]:
        for x in sorted(df[xcol].unique()):
            part = df[df[xcol] == x]
            for i, col in enumerate(METHODS):
                seed_values = part.groupby("seed")[col].mean().to_numpy()
                mean, lo, hi = bootstrap(seed_values, int(x) * 10 + i)
                rows.append(dict(sweep=sweep, x=x, method=col, mean_ms=mean,
                                 ci_low=lo, ci_high=hi, n_seeds=len(seed_values)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT)
    summary = summarize(df)
    summary.to_csv(SUMMARY, index=False)
    plt.rcParams.update({"font.family": "YuGothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)
    panels = [
        ("dimension", "次元数による計算時間の推移", "特徴数"),
        ("classes", "クラス数による計算時間の推移", "クラス数"),
    ]
    for ax, (sweep, title, xlabel) in zip(axes, panels):
        for method, (label, color, marker) in METHODS.items():
            part = summary[(summary.sweep == sweep) & (summary.method == method)]
            x = part.x.to_numpy()
            ax.plot(x, part.mean_ms, label=label, color=color, marker=marker,
                    linewidth=2, markersize=6)
            ax.fill_between(x, part.ci_low, part.ci_high, color=color, alpha=0.12)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("1説明あたりの時間（ms）")
        ax.grid(axis="y", color="#D1D5DB", alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xticks([8, 14, 20])
    axes[1].set_xticks(range(3, 11))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("サロゲートの学習時間", x=0.02, ha="left", fontsize=16, weight="bold")
    fig.text(0.98, 0.96, "特徴選択＋最終再学習・独立データ10シードの平均と95%信頼区間",
             ha="right", va="top", fontsize=9, color="#4B5563")
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
