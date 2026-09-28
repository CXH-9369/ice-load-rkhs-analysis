#!/usr/bin/env python3
"""Create paper-facing figures using raw specimens plus median/IQR summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from run_event_stage_bootstrap_analysis import (
    DELTA_METRICS,
    PAIR_LABELS,
    PAIR_ORDER,
    PRIMARY_METRICS,
    RATE_COLORS,
    RATE_LABELS,
    RATE_ORDER,
    SPATIAL_METRICS,
    STAGE_LABELS,
    STAGE_ORDER,
)


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    q25, median, q75 = np.quantile(finite, [0.25, 0.50, 0.75])
    return float(median), float(q25), float(q75)


def plot_state_trajectories(
    table: pd.DataFrame,
    metrics: dict[str, tuple[str, str, float]],
    stages: list[str],
    output: Path,
    title: str,
    subtitle: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    x = np.arange(len(stages), dtype=float)
    for axis, (metric, (label, unit, scale)) in zip(axes.flat, metrics.items()):
        for rate in RATE_ORDER:
            rate_table = table[table["rate_group"] == rate]
            for _, specimen_rows in rate_table.groupby("specimen_id"):
                specimen_rows = specimen_rows.copy()
                specimen_rows["stage_order"] = specimen_rows["stage"].astype(str).map(
                    {stage: index for index, stage in enumerate(stages)}
                )
                specimen_rows = specimen_rows.dropna(subset=["stage_order"]).sort_values("stage_order")
                axis.plot(
                    specimen_rows["stage_order"],
                    specimen_rows[metric].to_numpy(dtype=float) * scale,
                    color=RATE_COLORS[rate],
                    lw=0.8,
                    alpha=0.20,
                )
            medians, q25s, q75s = [], [], []
            for stage in stages:
                values = rate_table[rate_table["stage"].astype(str) == stage][metric].to_numpy(dtype=float) * scale
                median, q25, q75 = median_iqr(values)
                medians.append(median)
                q25s.append(q25)
                q75s.append(q75)
            axis.fill_between(x, q25s, q75s, color=RATE_COLORS[rate], alpha=0.14)
            axis.plot(
                x,
                medians,
                color=RATE_COLORS[rate],
                lw=2.2,
                marker="o",
                ms=5,
                label=RATE_LABELS[rate],
            )
        if metric.startswith("delta_"):
            axis.axhline(0.0, color="#4b5563", lw=0.9)
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(x, [STAGE_LABELS[stage] for stage in stages], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(0.5, 0.900, subtitle, ha="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.855, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def add_iqr_box(
    axis: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    width: float = 0.13,
) -> None:
    median, q25, q75 = median_iqr(values)
    rectangle = Rectangle(
        (x - width / 2.0, q25),
        width,
        q75 - q25,
        facecolor=color,
        edgecolor="none",
        alpha=0.20,
        zorder=2,
    )
    axis.add_patch(rectangle)
    axis.plot([x - width / 2.0, x + width / 2.0], [median, median], color=color, lw=2.3, zorder=4)


def plot_spatial_distributions(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    base = np.arange(len(PAIR_ORDER), dtype=float)
    offsets = {"low": -0.22, "medium": 0.0, "high": 0.22}
    rng = np.random.default_rng(20261001)
    for axis, (metric, (label, unit, scale)) in zip(axes.flat, SPATIAL_METRICS.items()):
        for rate in RATE_ORDER:
            subset = table[table["rate_group"] == rate]
            for pair_index, pair in enumerate(PAIR_ORDER):
                values = subset[subset["event_pair"].astype(str) == pair][metric].to_numpy(dtype=float) * scale
                x = base[pair_index] + offsets[rate]
                jitter = rng.uniform(-0.045, 0.045, size=len(values))
                axis.scatter(
                    np.full(len(values), x) + jitter,
                    values,
                    s=27,
                    color=RATE_COLORS[rate],
                    alpha=0.55,
                    edgecolors="white",
                    linewidths=0.35,
                    zorder=3,
                )
                add_iqr_box(axis, x, values, RATE_COLORS[rate])
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(base, [PAIR_LABELS[pair] for pair in PAIR_ORDER], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.20)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=RATE_COLORS[rate], markersize=7, label=RATE_LABELS[rate])
        for rate in RATE_ORDER
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Spatial redistribution between event-defined pressure fields", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.900,
        "Points are specimens; shaded rectangles show the interquartile range and horizontal bars show medians. No long whiskers are used.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.855, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_effect_matrix(effects: pd.DataFrame, output: Path) -> None:
    metrics = list(PRIMARY_METRICS)
    matrix = np.full((len(metrics), len(STAGE_ORDER)), np.nan)
    q_values = np.full_like(matrix, np.nan)
    for metric_index, metric in enumerate(metrics):
        for stage_index, stage in enumerate(STAGE_ORDER):
            row = effects[(effects["metric"] == metric) & (effects["stage"] == stage)].iloc[0]
            matrix[metric_index, stage_index] = float(row["cliffs_delta_high_vs_low"])
            q_values[metric_index, stage_index] = float(row["bh_q_cliffs_delta"])
    fig, axis = plt.subplots(figsize=(10, 6))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            symbol = "*" if q_values[row, column] <= 0.05 else ""
            axis.text(
                column,
                row,
                f"δ={matrix[row, column]:.2f}{symbol}\nq={q_values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if abs(matrix[row, column]) > 0.55 else "black",
            )
    axis.set_xticks(np.arange(len(STAGE_ORDER)), [STAGE_LABELS[stage] for stage in STAGE_ORDER], rotation=15, ha="right")
    axis.set_yticks(
        np.arange(len(metrics)),
        [PRIMARY_METRICS[metric][0] for metric in metrics],
    )
    axis.set_title("Low–high rate effect-size matrix at matched event states")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Cliff's δ (positive: high rate > low rate)")
    fig.text(
        0.5,
        0.02,
        "Asterisk: BH-adjusted exact permutation q≤0.05. Full bootstrap intervals remain in the supplementary forest plot.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.28, right=0.94, bottom=0.18, top=0.90)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_incremental_rkhs_iqr(table: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(11, 7))
    for rate in RATE_ORDER:
        subset = table[table["rate_group"] == rate]
        for _, specimen_rows in subset.groupby("specimen_id"):
            specimen_rows = specimen_rows.sort_values("force_fraction_of_specimen_peak")
            axis.plot(
                specimen_rows["force_fraction_of_specimen_peak"],
                specimen_rows["rkhs_distance_from_10pct_load_state"],
                color=RATE_COLORS[rate],
                lw=0.8,
                alpha=0.22,
            )
        grouped = subset.groupby("force_fraction_of_specimen_peak")["rkhs_distance_from_10pct_load_state"]
        x, medians, q25s, q75s = [], [], [], []
        for progress, values in grouped:
            median, q25, q75 = median_iqr(values.to_numpy(dtype=float))
            x.append(float(progress))
            medians.append(median)
            q25s.append(q25)
            q75s.append(q75)
        order = np.argsort(x)
        x = np.asarray(x)[order]
        medians = np.asarray(medians)[order]
        q25s = np.asarray(q25s)[order]
        q75s = np.asarray(q75s)[order]
        axis.fill_between(x, q25s, q75s, color=RATE_COLORS[rate], alpha=0.14)
        axis.plot(x, medians, color=RATE_COLORS[rate], lw=2.3, marker="o", ms=4, label=RATE_LABELS[rate])
    axis.set_xlabel("Machine force / specimen peak force")
    axis.set_ylabel("RKHS distance from the 10%-load pressure state")
    axis.set_title("Initial-state-adjusted nonlinear pressure evolution")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.5,
        0.02,
        "Thin lines show all specimens; shaded bands show the interquartile range, not 95% error bars.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1/academic"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/paper_figures"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bootstrap_root = args.root / "bootstrap_event_analysis"
    incremental_root = args.root / "incremental_rkhs_sensitivity"
    stage_table = pd.read_csv(bootstrap_root / "event_stage_pressure_metrics.csv")
    change_table = pd.read_csv(bootstrap_root / "baseline_adjusted_event_changes.csv")
    spatial_table = pd.read_csv(bootstrap_root / "event_pair_spatial_statistics.csv")
    effects = pd.read_csv(bootstrap_root / "low_vs_high_effect_sizes.csv")
    incremental = pd.read_csv(incremental_root / "incremental_rkhs_distance_trajectories.csv")

    plot_state_trajectories(
        stage_table,
        PRIMARY_METRICS,
        STAGE_ORDER,
        args.output / "paper_event_pressure_states_iqr.png",
        "Pressure-state evolution across event-defined states",
        "Thin lines are specimens; thick lines are medians; shaded bands are interquartile ranges. Event states are not always chronological.",
    )
    plot_state_trajectories(
        change_table,
        DELTA_METRICS,
        STAGE_ORDER[1:],
        args.output / "paper_baseline_adjusted_evolution_iqr.png",
        "Within-specimen pressure evolution relative to the pre-transition state",
        "Raw specimen changes, medians and interquartile ranges are shown; 95% bootstrap intervals are retained only in supplementary outputs.",
    )
    plot_spatial_distributions(
        spatial_table,
        args.output / "paper_spatial_redistribution_iqr.png",
    )
    plot_effect_matrix(effects, args.output / "paper_low_high_effect_matrix.png")
    plot_incremental_rkhs_iqr(
        incremental,
        args.output / "paper_incremental_rkhs_iqr.png",
    )
    policy = """# 论文主图不确定性表达规则\n\n- 主图使用全部独立试件、组中位数与四分位区间；不使用视觉上过长的95% Bootstrap误差棒。\n- n=3的低速组不使用核密度小提琴图，避免制造虚假的平滑分布；优先显示全部原始点或单试件轨迹。\n- 95% Bootstrap区间、Cliff's δ区间、精确p值和BH q值继续保留在补充图与CSV中，不删除、不截短。\n- 四分位区间是分布描述，不得在图注中称为置信区间或标准误。\n- 事件状态并非所有试件都按同一时间顺序出现；用连线只表示同一试件的语义状态对应，不宣称统一时间路径。\n"""
    (args.output / "paper_uncertainty_representation_policy_zh.md").write_text(policy, encoding="utf-8")
    replacements = pd.DataFrame(
        [
            {
                "supplementary_or_audit_figure": "bootstrap_event_analysis/event_stage_bootstrap_intervals.png",
                "paper_main_figure": "paper_event_pressure_states_iqr.png",
                "main_figure_representation": "all specimens + median + IQR",
            },
            {
                "supplementary_or_audit_figure": "bootstrap_event_analysis/baseline_adjusted_pressure_evolution.png",
                "paper_main_figure": "paper_baseline_adjusted_evolution_iqr.png",
                "main_figure_representation": "within-specimen changes + median + IQR",
            },
            {
                "supplementary_or_audit_figure": "bootstrap_event_analysis/pressure_field_redistribution_statistics.png",
                "paper_main_figure": "paper_spatial_redistribution_iqr.png",
                "main_figure_representation": "all specimen values + median + IQR box",
            },
            {
                "supplementary_or_audit_figure": "bootstrap_event_analysis/low_vs_high_effect_sizes.png",
                "paper_main_figure": "paper_low_high_effect_matrix.png",
                "main_figure_representation": "effect estimate heatmap + BH q-value",
            },
            {
                "supplementary_or_audit_figure": "incremental_rkhs_sensitivity/incremental_rkhs_distance_trajectories.png",
                "paper_main_figure": "paper_incremental_rkhs_iqr.png",
                "main_figure_representation": "all specimen trajectories + median + IQR",
            },
        ]
    )
    replacements.to_csv(args.output / "paper_figure_replacement_index.csv", index=False, encoding="utf-8-sig")
    print(f"Created 5 paper-facing figures in {args.output}")


if __name__ == "__main__":
    main()
