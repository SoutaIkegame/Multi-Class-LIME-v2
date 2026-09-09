"""Plot feature-role recovery and held-out fidelity for the synthetic study."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "feature_role_results.csv"
OUTPUT = RESULTS / "feature_role_comparison.png"
SUMMARY = RESULTS / "feature_role_summary.csv"

METHODS = {
    "ovr_top_class": ("クラス別LIME（予測1位）", "#9CA3AF", "--", "o"),
    "ovr": ("OVR比較（ちょうどK特徴）", "#4B5563", "-", "o"),
    "two_class_lime": ("2クラス選択＋通常LIME", "#2563EB", "-", "s"),
    "contrastive": ("対数比＋リッジ回帰", "#D97706", "-", "D"),
    "logistic": ("局所ロジスティック回帰", "#059669", "-", "^"),
}


def bootstrap(values, seed):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, (10_000, len(values)), replace=True).mean(axis=1)
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def summarize(df):
    rows = []
    metrics = ["pair_recall", "shared_selection_rate", "fidelity_test"]
    for strength in sorted(df.shared_strength.unique()):
        part = df[df.shared_strength == strength]
        for method in METHODS:
            for metric in metrics:
                col = f"{method}_{metric}"
                if col not in part:
                    continue
                seed_values = part.groupby("seed")[col].mean().to_numpy()
                mean, lo, hi = bootstrap(seed_values, int(100 * strength) + len(rows))
                rows.append(dict(shared_strength=strength, method=method, metric=metric,
                                 mean=mean, ci_low=lo, ci_high=hi,
                                 n_seeds=len(seed_values)))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT)
    summary = summarize(df)
    summary.to_csv(SUMMARY, index=False)
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12.5,
                         "font.family": "YuGothic", "axes.unicode_minus": False})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), constrained_layout=True)
    panels = [
        ("pair_recall", "A/B競合特徴の再現率", "高いほど良い", (0.72, 1.01)),
        ("shared_selection_rate", "A/B共通特徴の表示率", "低いほど良い", (-0.01, 0.27)),
        ("fidelity_test", "未使用摂動でのペア忠実度", "高いほど良い", (0.88, 1.003)),
    ]
    for ax, (metric, title, ylabel, ylim) in zip(axes, panels):
        for method, (label, color, linestyle, marker) in METHODS.items():
            part = summary[(summary.metric == metric) & (summary.method == method)]
            if part.empty:
                continue
            part = part.sort_values("shared_strength")
            x = part.shared_strength.to_numpy()
            ax.plot(x, part["mean"], label=label, color=color, linestyle=linestyle,
                    marker=marker, linewidth=2, markersize=5)
            ax.fill_between(x, part.ci_low, part.ci_high, color=color, alpha=0.10)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("AとBに共通する特徴の強さ")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 1, 3, 5])
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D1D5DB", alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("特徴の役割が既知の線形softmaxモデル", x=0.02,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.98, 0.96, "K=3・独立データ20シードの平均と95%信頼区間",
             ha="right", va="top", fontsize=9, color="#4B5563")
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
