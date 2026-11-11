"""Plot scratch-model validation accuracies with bootstrap confidence intervals.

Edit the ``RESULTS`` list below when new experiments are available, then run:

    python plot_scratch_accuracy_results.py

The default output is ``scratch_model_accuracy_ci.png`` next to this script.
Use ``--show`` to open the plot or ``--help`` to see the other options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Experiment data
# ---------------------------------------------------------------------------
# Percentages are stored as percentage points (for example, 82.61 means
# 82.61%, not 0.8261). Add, remove, or edit dictionaries as needed.
RESULTS = [
    {
        "dataset": "CIFAR-10",
        "bits": 6,
        "method": "DoReFa",
        "validation_loss": 0.5339,
        "accuracy": 82.6100,
        "bootstrap_min": 81.3700,
        "bootstrap_max": 83.6700,
        "ci_low": 81.9098,
        "ci_high": 83.3602,
    },
    {
        "dataset": "CIFAR-10",
        "bits": 6,
        "method": "LSQ",
        "validation_loss": 0.3920,
        "accuracy": 90.4800,
        "bootstrap_min": 89.5200,
        "bootstrap_max": 91.6300,
        "ci_low": 89.9100,
        "ci_high": 91.0700,
    },
    {
        "dataset": "CIFAR-10",
        "bits": 4,
        "method": "LSQ",
        "validation_loss": 0.3623,
        "accuracy": 89.5300,
        "bootstrap_min": 88.6500,
        "bootstrap_max": 90.6500,
        "ci_low": 88.9500,
        "ci_high": 90.1500,
    },
    {
        "dataset": "CIFAR-10",
        "bits": 2,
        "method": "LSQ",
        "validation_loss": 0.5149,
        "accuracy": 82.7200,
        "bootstrap_min": 81.1600,
        "bootstrap_max": 83.9300,
        "ci_low": 82.0000,
        "ci_high": 83.4600,
    },
    {
        "dataset": "GTSDB",
        "bits": 6,
        "method": "LSQ",
        "validation_loss": 0.1189,
        "accuracy": 97.2299,
        "bootstrap_min": 94.4598,
        "bootstrap_max": 99.4460,
        "ci_low": 95.5609,
        "ci_high": 98.6219,
    },
    {
        "dataset": "GTSDB",
        "bits": 4,
        "method": "LSQ",
        "validation_loss": 0.1822,
        "accuracy": 95.2909,
        "bootstrap_min": 91.4127,
        "bootstrap_max": 98.3379,
        "ci_low": 93.0748,
        "ci_high": 97.2299,
    },
    {
        "dataset": "GTSDB",
        "bits": 2,
        "method": "LSQ",
        "validation_loss": 0.3863,
        "accuracy": 90.5817,
        "bootstrap_min": 85.8726,
        "bootstrap_max": 95.5679,
        "ci_low": 87.2576,
        "ci_high": 93.3518,
    },
]


BIT_COLORS = {
    6: "#3B82F6",
    4: "#10B981",
    2: "#F59E0B",
}
GROUP_GAP = 0.8


def validate_results(results: list[dict]) -> None:
    """Fail early when an edited result contains inconsistent bounds."""

    required = {
        "dataset",
        "bits",
        "method",
        "validation_loss",
        "accuracy",
        "bootstrap_min",
        "bootstrap_max",
        "ci_low",
        "ci_high",
    }
    for index, result in enumerate(results):
        missing = required.difference(result)
        if missing:
            raise ValueError(f"RESULTS[{index}] is missing: {sorted(missing)}")
        if not result["ci_low"] <= result["accuracy"] <= result["ci_high"]:
            raise ValueError(
                f"RESULTS[{index}] accuracy must lie inside its confidence interval."
            )
        if not result["bootstrap_min"] <= result["ci_low"]:
            raise ValueError(f"RESULTS[{index}] bootstrap_min exceeds ci_low.")
        if not result["ci_high"] <= result["bootstrap_max"]:
            raise ValueError(f"RESULTS[{index}] ci_high exceeds bootstrap_max.")


def build_positions(results: list[dict]) -> tuple[list[float], list[tuple[str, float, float]]]:
    """Return bar positions and dataset-group spans."""

    positions: list[float] = []
    spans: list[tuple[str, float, float]] = []
    cursor = 0.0
    group_start = 0.0
    current_dataset = results[0]["dataset"]

    for index, result in enumerate(results):
        if result["dataset"] != current_dataset:
            spans.append((current_dataset, group_start, positions[-1]))
            cursor += GROUP_GAP
            group_start = cursor
            current_dataset = result["dataset"]
        positions.append(cursor)
        cursor += 1.0

    spans.append((current_dataset, group_start, positions[-1]))
    return positions, spans


def make_plot(
    results: list[dict],
    output_path: Path,
    *,
    y_min: float = 75.0,
    dpi: int = 220,
    annotate_loss: bool = False,
    show: bool = False,
) -> None:
    validate_results(results)
    positions, spans = build_positions(results)

    accuracies = [result["accuracy"] for result in results]
    lower_errors = [result["accuracy"] - result["ci_low"] for result in results]
    upper_errors = [result["ci_high"] - result["accuracy"] for result in results]
    colors = [BIT_COLORS.get(result["bits"], "#64748B") for result in results]
    labels = [
        f'{result["bits"]}-bit\n{result["method"]}'
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    bars = ax.bar(
        positions,
        accuracies,
        width=0.72,
        color=colors,
        edgecolor="#172033",
        linewidth=0.9,
        yerr=[lower_errors, upper_errors],
        error_kw={
            "ecolor": "#172033",
            "elinewidth": 1.5,
            "capsize": 5,
            "capthick": 1.5,
        },
        zorder=3,
    )

    # Hatch DoReFa bars so the quantizer is distinguishable from LSQ at 6 bits.
    for bar, result in zip(bars, results):
        if result["method"].lower() == "dorefa":
            bar.set_hatch("///")

    y_max = min(100.0, max(result["ci_high"] for result in results) + 3.2)
    if y_min >= y_max:
        raise ValueError(f"y_min ({y_min}) must be below the plot maximum ({y_max}).")
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Validation accuracy (%)", fontsize=18)
    ax.set_xticks(positions, labels)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(
        "Scratch-model validation accuracy (95% bootstrap confidence intervals)",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    # ax.text(
    #     0.5,
    #     1,
    #     "Error bars show 95% bootstrap confidence intervals",
    #     transform=ax.transAxes,
    #     ha="center",
    #     va="bottom",
    #     fontsize=15,
    #     color="#475569",
    # )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Exact values make the figure usable even when confidence intervals overlap.
    text_offset = (y_max - y_min) * 0.018
    for bar, result in zip(bars, results):
        value_text = f'{result["accuracy"]:.2f}%'
        if annotate_loss:
            value_text += f'\nloss {result["validation_loss"]:.4f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            result["ci_high"] + text_offset,
            value_text,
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="semibold",
        )

    # Dataset labels and a separator keep all results in one readable chart.
    for dataset, start, end in spans:
        ax.text(
            (start + end) / 2,
            -0.16,
            dataset,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="#334155",
            clip_on=False,
        )
    for left_span, right_span in zip(spans, spans[1:]):
        separator = (left_span[2] + right_span[1]) / 2
        ax.axvline(separator, color="#94A3B8", linewidth=1.0, alpha=0.65)

    legend_handles = [
        Patch(facecolor=BIT_COLORS[bits], edgecolor="#172033", label=f"{bits}-bit")
        for bits in (6, 4, 2)
    ]
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor="#172033",
            hatch="///",
            label="DoReFa",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncols=4,
        frameon=False,
        fontsize=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved plot to: {output_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_output = Path(__file__).with_name("scratch_model_accuracy_ci.pdf")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output image path (default: {default_output.name}).",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=75.0,
        help="Lower y-axis limit; use 0 for a non-truncated axis (default: 75).",
    )
    parser.add_argument("--dpi", type=int, default=220, help="Output DPI (default: 220).")
    parser.add_argument(
        "--annotate-loss",
        action="store_true",
        help="Print validation loss below each accuracy annotation.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        RESULTS,
        args.output,
        y_min=args.y_min,
        dpi=args.dpi,
        annotate_loss=args.annotate_loss,
        show=args.show,
    )
