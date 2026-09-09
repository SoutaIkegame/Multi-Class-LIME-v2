"""線形softmax黒箱における真のペア係数復元を発表用に可視化する。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "groundtruth_results.csv"
OUTPUT = ROOT / "results" / "groundtruth_recovery.png"

METHODS = [
    ("OVR\n（ちょうどK特徴）", "ovr_spearman", "#6E7B8E", "o"),
    ("2クラス選択＋\n通常LIME", "pairwise_lime_spearman", "#2474E5", "s"),
    ("対数比＋\nリッジ回帰", "contrastive_spearman", "#D97706", "D"),
    ("局所ロジスティック\n回帰", "ovo_logistic_spearman", "#079669", "^"),
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(20260909)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    df = pd.read_csv(INPUT)
    # 説明点の擬似反復を避け、各グリッドセル×独立seed内で先に平均する。
    units = df.groupby(["n_features", "n_classes", "seed"], as_index=False).mean(numeric_only=True)

    plt.rcParams.update({
        "font.family": "YuGothic",
        "font.size": 15,
        "axes.titlesize": 22,
        "axes.titleweight": "bold",
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })
    fig, ax = plt.subplots(figsize=(13.2, 6.2), dpi=180)
    x = np.arange(len(METHODS))

    for i, (label, col, color, marker) in enumerate(METHODS):
        values = units[col].to_numpy()
        mean = values.mean()
        lo, hi = bootstrap_ci(values)
        ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt=marker,
                    markersize=13, color=color, ecolor=color, elinewidth=3,
                    capsize=7, capthick=2.5)
        ax.text(i, mean + 0.0033, f"{mean:.4f}", ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=color)

    ax.set_title("線形softmax黒箱：真のペア係数方向をどれだけ復元できるか", loc="left", pad=18)
    ax.text(0.0, 1.015, "Spearman順位相関・高いほど良い・95%ブートストラップ信頼区間",
            transform=ax.transAxes, color="#667085", fontsize=14)
    ax.set_ylabel("真の係数とのSpearman順位相関")
    ax.set_xticks(x, [m[0] for m in METHODS])
    ax.set_ylim(0.955, 1.006)
    ax.set_yticks(np.arange(0.96, 1.001, 0.01))
    ax.grid(axis="y", color="#D9DEE7", linewidth=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, -0.19, "特徴数8/14/20 × クラス数3/4/5 × 独立データ20シード",
            transform=ax.transAxes, ha="center", color="#667085", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
