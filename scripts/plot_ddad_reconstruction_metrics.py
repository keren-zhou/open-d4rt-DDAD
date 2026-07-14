#!/usr/bin/env python3
"""Plot the core DDAD reconstruction metrics before and after training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opend4rt-matplotlib")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BEFORE = Path("output/ddad_reconstruction_eval_before/summary.json")
DEFAULT_AFTER = Path("output/ddad_reconstruction_eval_after/summary.json")
DEFAULT_OUTPUT = Path("output/ddad_reconstruction_comparison")
ERROR_METRICS = (
    ("xyz_epe_raw_m", "XYZ EPE (raw)", "m"),
    ("xyz_epe_global_m", "XYZ EPE (scale-aligned)", "m"),
    ("xyz_epe_sim3_m", "XYZ EPE (Sim3)", "m"),
    ("depth_mae_global_m", "Depth MAE (scale-aligned)", "m"),
    ("depth_abs_rel_raw", "Depth AbsRel (raw)", ""),
    ("depth_abs_rel_global", "Depth AbsRel (scale-aligned)", ""),
)

BEFORE_COLOR = "#B5B0AA"
AFTER_COLOR = "#0072B2"
IDEAL_COLOR = "#D55E00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot before/after DDAD validation reconstruction metrics."
    )
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--before-label", default="Before training")
    parser.add_argument("--after-label", default="After training")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary object in {path}")
    for key, _, _ in ERROR_METRICS:
        if key not in summary:
            raise ValueError(f"Missing metric {key!r} in {path}")
    if "scale_global" not in summary:
        raise ValueError(f"Missing metric 'scale_global' in {path}")
    return summary


def value_text(value: float, unit: str) -> str:
    if unit == "m":
        return f"{value:.3f} m"
    return f"{value:.4f}"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": "#D8D1C7",
            "axes.labelcolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.5,
            "grid.color": "#E7E5E4",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.65,
            "xtick.color": "#6B7280",
            "ytick.color": "#6B7280",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )


def main() -> int:
    args = parse_args()
    before = load_summary(args.before)
    after = load_summary(args.after)
    apply_style()

    fig, (ax_error, ax_scale) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.7),
        gridspec_kw={"width_ratios": [3.6, 1.4]},
    )

    y = np.arange(len(ERROR_METRICS), dtype=np.float64)
    bar_height = 0.31
    before_relative = np.full(len(ERROR_METRICS), 100.0)
    after_relative = np.asarray(
        [float(after[key]) / float(before[key]) * 100.0 for key, _, _ in ERROR_METRICS]
    )

    before_bars = ax_error.barh(
        y - bar_height / 2,
        before_relative,
        height=bar_height,
        color=BEFORE_COLOR,
        label=args.before_label,
    )
    after_bars = ax_error.barh(
        y + bar_height / 2,
        after_relative,
        height=bar_height,
        color=AFTER_COLOR,
        label=args.after_label,
    )
    ax_error.set_title("(a) Reconstruction error", loc="left", pad=12)
    ax_error.set_yticks(y, [label for _, label, _ in ERROR_METRICS])
    ax_error.invert_yaxis()
    ax_error.set_xlim(0.0, 148.0)
    ax_error.set_xticks([0, 25, 50, 75, 100])
    ax_error.set_xlabel("Relative error (% of pre-training; lower is better)")
    ax_error.grid(axis="y", visible=False)
    ax_error.axvline(100.0, color="#78716C", linewidth=0.8, linestyle=":")

    for index, (key, _, unit) in enumerate(ERROR_METRICS):
        before_value = float(before[key])
        after_value = float(after[key])
        reduction = (before_value - after_value) / before_value * 100.0
        ax_error.text(
            after_relative[index] - 1.5,
            y[index] + bar_height / 2,
            f"-{reduction:.1f}%",
            ha="right",
            va="center",
            fontsize=8.2,
            color="white",
            fontweight="bold",
        )
        ax_error.text(
            104.0,
            y[index],
            f"{value_text(before_value, unit)}  ->  {value_text(after_value, unit)}",
            ha="left",
            va="center",
            fontsize=8.2,
            color="#4B5563",
        )

    before_scale = float(before["scale_global"])
    after_scale = float(after["scale_global"])
    before_distance = abs(before_scale - 1.0)
    after_distance = abs(after_scale - 1.0)
    distance_reduction = (
        (before_distance - after_distance) / before_distance * 100.0
        if before_distance > 0.0
        else 0.0
    )

    ax_scale.set_title("(b) Scale calibration", loc="left", pad=12)
    ax_scale.axvline(1.0, color=IDEAL_COLOR, linewidth=1.2, linestyle="--")
    ax_scale.annotate(
        "",
        xy=(after_scale, 0.0),
        xytext=(before_scale, 0.0),
        arrowprops={"arrowstyle": "-|>", "color": "#7C7873", "linewidth": 1.5},
    )
    ax_scale.scatter(before_scale, 0.0, s=78, color=BEFORE_COLOR, zorder=3)
    ax_scale.scatter(after_scale, 0.0, s=78, color=AFTER_COLOR, zorder=3)
    ax_scale.text(
        before_scale,
        0.085,
        f"{args.before_label}\n{before_scale:.3f}x",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4B5563",
    )
    ax_scale.text(
        after_scale,
        -0.085,
        f"{args.after_label}\n{after_scale:.3f}x",
        ha="center",
        va="top",
        fontsize=8.5,
        color=AFTER_COLOR,
        fontweight="semibold",
    )
    ax_scale.text(
        1.0,
        0.31,
        "Ideal = 1",
        ha="left",
        va="center",
        fontsize=8.5,
        color=IDEAL_COLOR,
    )
    ax_scale.text(
        0.5,
        0.08,
        f"{distance_reduction:.1f}% closer\nto metric scale",
        transform=ax_scale.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=AFTER_COLOR,
        fontweight="bold",
    )
    ax_scale.set_xlim(0.0, max(before_scale, after_scale) * 1.1)
    ax_scale.set_ylim(-0.42, 0.42)
    ax_scale.set_xlabel("Scale multiplier")
    ax_scale.set_yticks([])
    ax_scale.grid(axis="y", visible=False)
    ax_scale.spines["left"].set_visible(False)

    scenes = int(after.get("num_scenes", 0))
    queries = int(after.get("total_queries", 0))
    fig.suptitle(
        "DDAD validation reconstruction",
        x=0.06,
        y=0.98,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.06,
        0.925,
        f"Before vs. after training  |  {scenes} validation scenes  |  {queries:,} sparse LiDAR queries",
        ha="left",
        color="#6B7280",
        fontsize=9,
    )
    fig.legend(
        [before_bars[0], after_bars[0]],
        [args.before_label, args.after_label],
        loc="upper right",
        bbox_to_anchor=(0.96, 0.965),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.24, right=0.97, top=0.82, bottom=0.17, wspace=0.34)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / "ddad_reconstruction_metrics.png"
    pdf_path = args.output_dir / "ddad_reconstruction_metrics.pdf"
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
