"""Plot the exact-K explainer comparison across black-box structures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INPUT = RESULTS / "blackbox_comparison_results.csv"
OUTPUT = RESULTS / "blackbox_comparison.png"
SUMMARY = RESULTS / "blackbox_comparison_summary.csv"

BLACKBOXES = ["linear_softmax", "nonlinear_softmax", "random_forest"]
BLACKBOX_LABELS = {
    "linear_softmax": "線形\nsoftmax",
    "nonlinear_softmax": "非線形NN\n＋softmax",
    "random_forest": "ランダム\nフォレスト",
}
METHODS = {
    "ovr": {
        "label": "OVR（ちょうどK特徴）",
        "fidelity": "ovr_fidelity_test",
        "color": "#6B7280",
        "marker": "o",
    },
    "two_class_lime": {
        "label": "2クラス選択＋通常LIME",
        "fidelity": "two_class_lime_fidelity_test",
        "brier": "two_class_lime_brier_test",
        "color": "#2563EB",
        "marker": "s",
    },
    "contrastive": {
        "label": "対数比＋リッジ回帰",
        "fidelity": "contrastive_fidelity_test",
        "brier": "contrastive_brier_test",
        "color": "#D97706",
        "marker": "D",
    },
    "logistic": {
        "label": "局所ロジスティック回帰",
        "fidelity": "logistic_fidelity_test",
        "brier": "logistic_brier_test",
        "color": "#059669",
        "marker": "^",
    },
}


def seed_bootstrap_ci(values: np.ndarray, seed: int = 20260909) -> tuple[float, float]:
    """Percentile CI over independent dataset-seed means."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for blackbox in BLACKBOXES:
        block = df[df["blackbox"] == blackbox]
        for method, spec in METHODS.items():
            for metric in ("fidelity", "brier"):
                column = spec.get(metric)
                if column is None:
                    continue
                # First average repeated instances and grid/K conditions within
                # each independent dataset seed; seeds are the inference units.
                seed_values = block.groupby("seed")[column].mean().to_numpy()
                low, high = seed_bootstrap_ci(seed_values)
                rows.append(
                    {
                        "blackbox": blackbox,
                        "method": method,
                        "metric": metric,
                        "mean": seed_values.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "n_seeds": len(seed_values),
                    }
                )
    return pd.DataFrame(rows)


def draw(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "YuGothic",
            "axes.unicode_minus": False,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    x = np.arange(len(BLACKBOXES))

    panels = [
        (axes[0], "fidelity", "未使用摂動でのペア符号忠実度", "高いほど良い"),
        (axes[1], "brier", "未使用摂動での重み付きBrier", "低いほど良い"),
    ]
    for ax, metric, title, direction in panels:
        method_keys = [key for key, spec in METHODS.items() if metric in spec]
        offsets = np.linspace(-0.24, 0.24, len(method_keys))
        for offset, method in zip(offsets, method_keys):
            spec = METHODS[method]
            part = (
                summary[(summary["metric"] == metric) & (summary["method"] == method)]
                .set_index("blackbox")
                .loc[BLACKBOXES]
            )
            means = part["mean"].to_numpy()
            yerr = np.vstack(
                [means - part["ci_low"].to_numpy(), part["ci_high"].to_numpy() - means]
            )
            ax.errorbar(
                x + offset,
                means,
                yerr=yerr,
                fmt=spec["marker"],
                markersize=7,
                capsize=3,
                linewidth=1.7,
                color=spec["color"],
                label=spec["label"],
            )
        ax.set_xticks(x, [BLACKBOX_LABELS[b] for b in BLACKBOXES])
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel(direction)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.8, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0.74, 0.89)
    axes[1].set_ylim(0.0, 0.078)
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = axes[1].get_legend_handles_labels()
    legend = fig.legend(
        handles + handles2[1:],
        labels + labels2[1:],
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    # Remove duplicate pair-method entries while keeping OVR from the left panel.
    unique = dict(zip(labels + labels2, handles + handles2))
    legend.remove()
    fig.legend(
        unique.values(), unique.keys(), loc="outside lower center", ncol=4, frameon=False
    )
    fig.suptitle(
        "ブラックボックス構造別の比較（表示特徴数K）",
        fontsize=16,
        weight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.98,
        0.965,
        "独立データ20シードの平均と95%ブートストラップ信頼区間",
        ha="right",
        va="top",
        fontsize=9,
        color="#4B5563",
    )
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")


def main() -> None:
    df = pd.read_csv(INPUT)
    summary = summarize(df)
    summary.to_csv(SUMMARY, index=False)
    draw(summary)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
