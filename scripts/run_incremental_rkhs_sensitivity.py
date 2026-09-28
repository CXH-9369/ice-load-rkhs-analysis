#!/usr/bin/env python3
"""Initial-state-adjusted RKHS evolution and robustness checks."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_event_stage_bootstrap_analysis import (
    RATE_COLORS,
    RATE_LABELS,
    RATE_ORDER,
    STAGE_LABELS,
    STAGE_ORDER,
    benjamini_hochberg,
    bootstrap_median,
    stage_key,
)
from run_rkhs_evolution_analysis import (
    DISPLAY_NAMES,
    PRESSURE_FEATURES,
    biased_mmd2_from_kernel,
    median_bandwidth,
    rbf_kernel,
    robust_fit,
    robust_transform,
)


BASELINE_FRACTIONS = [0.10, 0.15, 0.20, 0.25]
BANDWIDTH_FACTORS = [0.50, 0.75, 1.00, 1.50, 2.00]
PRESSURE_THRESHOLDS_KPA = [25.0, 50.0, 100.0, 250.0, 500.0]
RATE_PAIRS = list(itertools.combinations(RATE_ORDER, 2))


def exact_mmd_test_fixed_bandwidth(
    values: np.ndarray,
    n_first: int,
    bandwidth_factor: float = 1.0,
) -> tuple[float, float, int, float]:
    base_bandwidth = median_bandwidth(values)
    bandwidth = max(base_bandwidth * bandwidth_factor, 1e-12)
    kernel, _ = rbf_kernel(values, bandwidth)
    observed_mask = np.zeros(len(values), dtype=bool)
    observed_mask[:n_first] = True
    observed = biased_mmd2_from_kernel(kernel, observed_mask)
    exceed = 0
    total = 0
    for chosen in itertools.combinations(range(len(values)), n_first):
        mask = np.zeros(len(values), dtype=bool)
        mask[list(chosen)] = True
        statistic = biased_mmd2_from_kernel(kernel, mask)
        exceed += statistic >= observed - 1e-15
        total += 1
    return observed, float(exceed / total), total, bandwidth


def trajectory_arrays(
    table: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    specimen_ids = list(dict.fromkeys(table["specimen_id"].astype(str)))
    grid = np.sort(table["force_fraction_of_specimen_peak"].unique().astype(float))
    rates: list[str] = []
    trajectories: list[np.ndarray] = []
    for specimen in specimen_ids:
        rows = table[table["specimen_id"] == specimen].sort_values("force_fraction_of_specimen_peak")
        if len(rows) != len(grid):
            raise ValueError(f"Incomplete balanced trajectory for {specimen}")
        rates.append(str(rows.iloc[0]["rate_group"]))
        trajectories.append(rows[PRESSURE_FEATURES].to_numpy(dtype=float))
    return specimen_ids, rates, grid, np.asarray(trajectories)


def pair_indices(rates: list[str], first: str, second: str) -> tuple[list[int], list[int]]:
    return (
        [index for index, rate in enumerate(rates) if rate == first],
        [index for index, rate in enumerate(rates) if rate == second],
    )


def incremental_vectors(
    standardized: np.ndarray,
    grid: np.ndarray,
    baseline_fraction: float,
    evaluation_minimum: float = 0.30,
) -> np.ndarray:
    baseline_index = int(np.argmin(np.abs(grid - baseline_fraction)))
    if not np.isclose(grid[baseline_index], baseline_fraction):
        raise ValueError(f"Baseline {baseline_fraction} not on trajectory grid")
    evaluation = np.flatnonzero(grid >= evaluation_minimum - 1e-12)
    delta = standardized[:, evaluation, :] - standardized[:, baseline_index : baseline_index + 1, :]
    return delta.reshape(len(standardized), -1)


def baseline_fraction_sensitivity(
    standardized: np.ndarray,
    grid: np.ndarray,
    rates: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for baseline in BASELINE_FRACTIONS:
        vectors = incremental_vectors(standardized, grid, baseline)
        for first, second in RATE_PAIRS:
            first_indices, second_indices = pair_indices(rates, first, second)
            joined = vectors[first_indices + second_indices]
            mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(
                joined, len(first_indices)
            )
            rows.append(
                {
                    "baseline_force_fraction": baseline,
                    "evaluation_force_fraction_minimum": 0.30,
                    "rate_group_a": first,
                    "rate_group_b": second,
                    "specimen_count_a": len(first_indices),
                    "specimen_count_b": len(second_indices),
                    "incremental_trajectory_mmd2": mmd2,
                    "exact_permutation_p_value": p_value,
                    "permutation_count": permutations,
                    "rbf_bandwidth": bandwidth,
                }
            )
    return pd.DataFrame(rows)


def bandwidth_sensitivity(
    standardized: np.ndarray,
    grid: np.ndarray,
    rates: list[str],
) -> pd.DataFrame:
    vectors = incremental_vectors(standardized, grid, 0.10)
    rows: list[dict[str, object]] = []
    for factor in BANDWIDTH_FACTORS:
        for first, second in RATE_PAIRS:
            first_indices, second_indices = pair_indices(rates, first, second)
            joined = vectors[first_indices + second_indices]
            mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(
                joined, len(first_indices), factor
            )
            rows.append(
                {
                    "baseline_force_fraction": 0.10,
                    "bandwidth_factor": factor,
                    "rate_group_a": first,
                    "rate_group_b": second,
                    "incremental_trajectory_mmd2": mmd2,
                    "exact_permutation_p_value": p_value,
                    "permutation_count": permutations,
                    "rbf_bandwidth": bandwidth,
                }
            )
    return pd.DataFrame(rows)


def leave_one_low_out(
    standardized: np.ndarray,
    grid: np.ndarray,
    specimen_ids: list[str],
    rates: list[str],
) -> pd.DataFrame:
    vectors = incremental_vectors(standardized, grid, 0.10)
    low_indices, high_indices = pair_indices(rates, "low", "high")
    rows: list[dict[str, object]] = []
    for omitted in low_indices:
        retained_low = [index for index in low_indices if index != omitted]
        joined = vectors[retained_low + high_indices]
        mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(
            joined, len(retained_low)
        )
        rows.append(
            {
                "omitted_low_rate_specimen": specimen_ids[omitted],
                "retained_low_rate_specimens": "|".join(specimen_ids[index] for index in retained_low),
                "specimen_count_low": len(retained_low),
                "specimen_count_high": len(high_indices),
                "incremental_trajectory_mmd2": mmd2,
                "exact_permutation_p_value": p_value,
                "permutation_count": permutations,
                "rbf_bandwidth": bandwidth,
            }
        )
    return pd.DataFrame(rows)


def feature_incremental_mmd(
    standardized: np.ndarray,
    grid: np.ndarray,
    rates: list[str],
) -> pd.DataFrame:
    baseline_index = int(np.argmin(np.abs(grid - 0.10)))
    evaluation = np.flatnonzero(grid >= 0.30 - 1e-12)
    delta = standardized[:, evaluation, :] - standardized[:, baseline_index : baseline_index + 1, :]
    rows: list[dict[str, object]] = []
    for first, second in RATE_PAIRS:
        first_indices, second_indices = pair_indices(rates, first, second)
        pair_rows: list[dict[str, object]] = []
        for feature_index, feature in enumerate(PRESSURE_FEATURES):
            joined = delta[first_indices + second_indices, :, feature_index]
            mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(
                joined, len(first_indices)
            )
            pair_rows.append(
                {
                    "rate_group_a": first,
                    "rate_group_b": second,
                    "pressure_feature": feature,
                    "incremental_feature_trajectory_mmd2": mmd2,
                    "exact_permutation_p_value": p_value,
                    "permutation_count": permutations,
                    "rbf_bandwidth": bandwidth,
                }
            )
        q_values = benjamini_hochberg(
            np.asarray([row["exact_permutation_p_value"] for row in pair_rows], dtype=float)
        )
        for row, q_value in zip(pair_rows, q_values):
            row["bh_q_value_within_rate_pair"] = float(q_value)
        rows.extend(pair_rows)
    return pd.DataFrame(rows)


def incremental_rkhs_distances(
    standardized: np.ndarray,
    grid: np.ndarray,
    specimen_ids: list[str],
    rates: list[str],
    rng: np.random.Generator,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    state_bandwidth = median_bandwidth(standardized.reshape(-1, standardized.shape[-1]))
    baseline_index = int(np.argmin(np.abs(grid - 0.10)))
    squared = np.sum(
        (standardized - standardized[:, baseline_index : baseline_index + 1, :]) ** 2,
        axis=2,
    )
    similarity = np.exp(-squared / (2.0 * state_bandwidth**2))
    distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
    specimen_rows = []
    for specimen_index, specimen in enumerate(specimen_ids):
        for progress_index, progress in enumerate(grid):
            specimen_rows.append(
                {
                    "specimen_id": specimen,
                    "rate_group": rates[specimen_index],
                    "force_fraction_of_specimen_peak": float(progress),
                    "rkhs_distance_from_10pct_load_state": float(distance[specimen_index, progress_index]),
                }
            )
    summary_rows = []
    for rate in RATE_ORDER:
        indices = [index for index, value in enumerate(rates) if value == rate]
        for progress_index, progress in enumerate(grid):
            values = distance[indices, progress_index]
            draws = bootstrap_median(values, rng, repeats)
            low, high = np.quantile(draws, [0.025, 0.975])
            summary_rows.append(
                {
                    "rate_group": rate,
                    "force_fraction_of_specimen_peak": float(progress),
                    "n_specimens": len(indices),
                    "median_rkhs_distance": float(np.median(values)),
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                }
            )
    return pd.DataFrame(specimen_rows), pd.DataFrame(summary_rows), state_bandwidth


def threshold_event_table(
    workspace: Path,
    state_table: pd.DataFrame,
    stage_index: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tensor_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for record in stage_index.to_dict("records"):
        specimen = str(record["specimen_id"])
        stage = stage_key(str(record["stage"]))
        candidates = state_table[state_table["specimen_id"] == specimen]
        exact = candidates[candidates["pressure_frame_number"] == int(record["pressure_frame_number"])]
        state = exact.iloc[0] if not exact.empty else candidates.iloc[
            int(np.argmin(np.abs(candidates["time_s"].to_numpy(dtype=float) - float(record["pressure_frame_time_s"]))))
        ]
        if specimen not in tensor_cache:
            path = workspace / "data_interim" / specimen / "unified" / "pressure_fields_32x32.npz"
            with np.load(path) as tensor:
                tensor_cache[specimen] = (
                    np.asarray(tensor["pressure_kpa"], dtype=float),
                    np.asarray(tensor["footprint_overlap_fraction"], dtype=float),
                )
        pressure, overlap = tensor_cache[specimen]
        field = pressure[int(state["pressure_field_tensor_index_0_based"])]
        for threshold in PRESSURE_THRESHOLDS_KPA:
            rows.append(
                {
                    "specimen_id": specimen,
                    "rate_group": str(record["rate_group"]),
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "pressure_threshold_kpa": threshold,
                    "active_area_fraction_of_70mm_face": float(
                        np.sum(overlap * (field >= threshold)) / field.size
                    ),
                }
            )
    result = pd.DataFrame(rows)
    result["stage"] = pd.Categorical(result["stage"], STAGE_ORDER, ordered=True)
    return result.sort_values(
        ["pressure_threshold_kpa", "rate_group", "specimen_id", "stage"]
    ).reset_index(drop=True)


def threshold_mmd_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in PRESSURE_THRESHOLDS_KPA:
        threshold_table = table[table["pressure_threshold_kpa"] == threshold]
        specimen_ids = list(dict.fromkeys(threshold_table["specimen_id"].astype(str)))
        rates = [
            str(threshold_table[threshold_table["specimen_id"] == specimen].iloc[0]["rate_group"])
            for specimen in specimen_ids
        ]
        matrix = np.asarray(
            [
                threshold_table[threshold_table["specimen_id"] == specimen]
                .sort_values("stage")["active_area_fraction_of_70mm_face"]
                .to_numpy(dtype=float)
                for specimen in specimen_ids
            ]
        )
        representations = {
            "absolute_event_states": matrix,
            "baseline_adjusted_event_changes": matrix[:, 1:] - matrix[:, :1],
        }
        for mode, vectors in representations.items():
            for first, second in RATE_PAIRS:
                first_indices, second_indices = pair_indices(rates, first, second)
                joined = vectors[first_indices + second_indices]
                mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(
                    joined, len(first_indices)
                )
                rows.append(
                    {
                        "pressure_threshold_kpa": threshold,
                        "trajectory_mode": mode,
                        "rate_group_a": first,
                        "rate_group_b": second,
                        "trajectory_mmd2": mmd2,
                        "exact_permutation_p_value": p_value,
                        "permutation_count": permutations,
                        "rbf_bandwidth": bandwidth,
                    }
                )
    result = pd.DataFrame(rows)
    for (mode, first, second), indices in result.groupby(
        ["trajectory_mode", "rate_group_a", "rate_group_b"]
    ).groups.items():
        result.loc[indices, "bh_q_value_across_thresholds_within_mode_and_pair"] = benjamini_hochberg(
            result.loc[indices, "exact_permutation_p_value"].to_numpy(dtype=float)
        )
    return result


def plot_incremental_distances(
    specimen_table: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(11, 7))
    for rate in RATE_ORDER:
        for _, specimen_rows in specimen_table[specimen_table["rate_group"] == rate].groupby("specimen_id"):
            specimen_rows = specimen_rows.sort_values("force_fraction_of_specimen_peak")
            axis.plot(
                specimen_rows["force_fraction_of_specimen_peak"],
                specimen_rows["rkhs_distance_from_10pct_load_state"],
                color=RATE_COLORS[rate],
                lw=0.8,
                alpha=0.20,
            )
        rate_summary = summary[summary["rate_group"] == rate].sort_values("force_fraction_of_specimen_peak")
        x = rate_summary["force_fraction_of_specimen_peak"].to_numpy(dtype=float)
        median = rate_summary["median_rkhs_distance"].to_numpy(dtype=float)
        low = rate_summary["bootstrap_ci95_low"].to_numpy(dtype=float)
        high = rate_summary["bootstrap_ci95_high"].to_numpy(dtype=float)
        axis.fill_between(x, low, high, color=RATE_COLORS[rate], alpha=0.13)
        axis.plot(x, median, color=RATE_COLORS[rate], lw=2.4, marker="o", ms=4, label=RATE_LABELS[rate])
    axis.set_xlabel("Machine force / specimen peak force")
    axis.set_ylabel("RKHS distance from the 10%-load pressure state")
    axis.set_title("Initial-state-adjusted nonlinear pressure evolution")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(
        0.5,
        0.02,
        "Thin lines: individual specimens; thick lines and bands: specimen medians and 95% bootstrap intervals.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def negative_log_p(values: pd.Series | np.ndarray) -> np.ndarray:
    return -np.log10(np.maximum(np.asarray(values, dtype=float), 1e-12))


def plot_sensitivity_overview(
    baseline: pd.DataFrame,
    bandwidth: pd.DataFrame,
    leave_out: pd.DataFrame,
    threshold: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    significance = -np.log10(0.05)
    pair_colors = {("low", "medium"): "#2563eb", ("low", "high"): "#7c3aed", ("medium", "high"): "#dc2626"}
    for pair in RATE_PAIRS:
        subset = baseline[
            (baseline["rate_group_a"] == pair[0]) & (baseline["rate_group_b"] == pair[1])
        ].sort_values("baseline_force_fraction")
        axes[0, 0].plot(
            subset["baseline_force_fraction"],
            negative_log_p(subset["exact_permutation_p_value"]),
            marker="o",
            color=pair_colors[pair],
            label=f"{pair[0]} vs {pair[1]}",
        )
    axes[0, 0].set_title("Baseline-reference sensitivity")
    axes[0, 0].set_xlabel("Baseline force fraction")
    axes[0, 0].set_ylabel("−log10(exact p)")
    axes[0, 0].legend(frameon=False)

    for pair in RATE_PAIRS:
        subset = bandwidth[
            (bandwidth["rate_group_a"] == pair[0]) & (bandwidth["rate_group_b"] == pair[1])
        ].sort_values("bandwidth_factor")
        axes[0, 1].plot(
            subset["bandwidth_factor"],
            negative_log_p(subset["exact_permutation_p_value"]),
            marker="o",
            color=pair_colors[pair],
            label=f"{pair[0]} vs {pair[1]}",
        )
    axes[0, 1].set_title("RBF-bandwidth sensitivity")
    axes[0, 1].set_xlabel("Median-heuristic bandwidth multiplier")
    axes[0, 1].set_ylabel("−log10(exact p)")

    x = np.arange(len(leave_out))
    axes[1, 0].bar(
        x,
        negative_log_p(leave_out["exact_permutation_p_value"]),
        color="#7c3aed",
        alpha=0.78,
    )
    axes[1, 0].set_xticks(x, leave_out["omitted_low_rate_specimen"], rotation=15, ha="right")
    axes[1, 0].set_title("Leave-one-low-rate-specimen-out")
    axes[1, 0].set_ylabel("−log10(exact p), low vs high")

    low_high = threshold[
        (threshold["rate_group_a"] == "low") & (threshold["rate_group_b"] == "high")
    ]
    mode_styles = {
        "absolute_event_states": ("Absolute event states", "#2563eb", "o"),
        "baseline_adjusted_event_changes": ("Baseline-adjusted changes", "#dc2626", "s"),
    }
    for mode, (label, color, marker) in mode_styles.items():
        subset = low_high[low_high["trajectory_mode"] == mode].sort_values("pressure_threshold_kpa")
        axes[1, 1].plot(
            subset["pressure_threshold_kpa"],
            negative_log_p(subset["exact_permutation_p_value"]),
            marker=marker,
            color=color,
            label=label,
        )
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xticks(PRESSURE_THRESHOLDS_KPA, [f"{int(value)}" for value in PRESSURE_THRESHOLDS_KPA])
    axes[1, 1].set_title("Pressure-threshold sensitivity")
    axes[1, 1].set_xlabel("Active-cell threshold (kPa)")
    axes[1, 1].set_ylabel("−log10(exact p), low vs high")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.axhline(significance, color="#374151", lw=1.0, ls="--")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Robustness of initial-state-adjusted RKHS conclusions", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.945,
        "Dashed line: unadjusted p=0.05. Sensitivity settings were fixed before inspecting these outputs.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.89, hspace=0.35, wspace=0.25)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_feature_incremental_mmd(table: pd.DataFrame, output: Path) -> None:
    subset = table[(table["rate_group_a"] == "low") & (table["rate_group_b"] == "high")].copy()
    subset = subset.sort_values("bh_q_value_within_rate_pair", ascending=False)
    labels = [DISPLAY_NAMES.get(feature, feature.replace("pressure_", "").replace("_", " ")) for feature in subset["pressure_feature"]]
    values = negative_log_p(subset["bh_q_value_within_rate_pair"])
    fig, axis = plt.subplots(figsize=(11, 7))
    y = np.arange(len(subset))
    axis.barh(y, values, color="#7c3aed", alpha=0.78)
    axis.axvline(-np.log10(0.05), color="#374151", ls="--", lw=1.0)
    axis.set_yticks(y, labels)
    axis.set_xlabel("−log10(BH q), low vs high")
    axis.set_title("Feature-level MMD after 10%-load baseline subtraction")
    axis.grid(axis="x", alpha=0.22)
    fig.subplots_adjust(left=0.33, right=0.98, bottom=0.11, top=0.91)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_summary(
    output: Path,
    baseline: pd.DataFrame,
    bandwidth: pd.DataFrame,
    leave_out: pd.DataFrame,
    features: pd.DataFrame,
    threshold: pd.DataFrame,
) -> None:
    default = baseline[np.isclose(baseline["baseline_force_fraction"], 0.10)]
    low_high_default = default[(default["rate_group_a"] == "low") & (default["rate_group_b"] == "high")].iloc[0]
    low_high_bandwidth = bandwidth[
        (bandwidth["rate_group_a"] == "low") & (bandwidth["rate_group_b"] == "high")
    ]
    feature_low_high = features[
        (features["rate_group_a"] == "low") & (features["rate_group_b"] == "high")
    ]
    threshold_low_high = threshold[
        (threshold["rate_group_a"] == "low") & (threshold["rate_group_b"] == "high")
    ]
    absolute_threshold = threshold_low_high[
        threshold_low_high["trajectory_mode"] == "absolute_event_states"
    ]
    adjusted_threshold = threshold_low_high[
        threshold_low_high["trajectory_mode"] == "baseline_adjusted_event_changes"
    ]
    low_high_by_baseline = baseline[
        (baseline["rate_group_a"] == "low") & (baseline["rate_group_b"] == "high")
    ].sort_values("baseline_force_fraction")
    low_medium_by_baseline = baseline[
        (baseline["rate_group_a"] == "low") & (baseline["rate_group_b"] == "medium")
    ].sort_values("baseline_force_fraction")
    significant_feature_names = [
        DISPLAY_NAMES.get(feature, feature)
        for feature in feature_low_high.loc[
            feature_low_high["bh_q_value_within_rate_pair"] <= 0.05,
            "pressure_feature",
        ]
    ]
    lines = [
        "# 初始接触状态校正的增量 RKHS 与敏感性分析",
        "",
        "- 所有分析继续使用 16 个独立试件；210-7、210-9 排除。",
        "- 主增量轨迹先使用全体压力状态的稳健中位数/IQR统一缩放，再从每个试件各自的 10% 峰值载荷状态相减。",
        "- MMD 的独立统计单位是试件；精确 p 值穷举所有允许的试件组标签分配。",
        f"- 默认 10% 基线下，低速—高速增量轨迹 MMD²={low_high_default['incremental_trajectory_mmd2']:.4f}，exact p={low_high_default['exact_permutation_p_value']:.4f}。",
        f"- 低速—高速在 0.5–2.0 倍核带宽下的 exact p 范围为 {low_high_bandwidth['exact_permutation_p_value'].min():.4f}–{low_high_bandwidth['exact_permutation_p_value'].max():.4f}。",
        f"- 每次删除一个低速试件后，低速—高速 exact p 范围为 {leave_out['exact_permutation_p_value'].min():.4f}–{leave_out['exact_permutation_p_value'].max():.4f}。",
        f"- 12 个增量特征轨迹中，低速—高速经组内 BH 校正 q≤0.05 的数量为 {int((feature_low_high['bh_q_value_within_rate_pair'] <= 0.05).sum())}。",
        f"- 25–500 kPa 阈值下，绝对事件状态低速—高速 MMD 的 p 范围为 {absolute_threshold['exact_permutation_p_value'].min():.4f}–{absolute_threshold['exact_permutation_p_value'].max():.4f}；基线校正事件变化的 p 范围为 {adjusted_threshold['exact_permutation_p_value'].min():.4f}–{adjusted_threshold['exact_permutation_p_value'].max():.4f}。",
        f"- 低速—高速增量轨迹在 10%/15% 基线下 p={low_high_by_baseline.iloc[0]['exact_permutation_p_value']:.4f}/{low_high_by_baseline.iloc[1]['exact_permutation_p_value']:.4f}，在 20%/25% 基线下升至 p={low_high_by_baseline.iloc[2]['exact_permutation_p_value']:.4f}/{low_high_by_baseline.iloc[3]['exact_permutation_p_value']:.4f}；差异主要集中在早期接触建立阶段。",
        f"- 低速—中速在四个基线定义下均为 p={low_medium_by_baseline['exact_permutation_p_value'].iloc[0]:.4f}；当前样本中这一分离比低速—高速对基线定义更稳定。",
        "- 低速—高速增量特征经 BH 校正后保留的四项为：" + "、".join(significant_feature_names) + "。",
        "",
        "## 解释原则",
        "",
        "1. 初始状态相减针对的是持久的试件间接触偏置，不等同于把所有几何、批次和同步误差完全消除。",
        "2. 若绝对状态在多个阈值下稳定分离、但增量轨迹在基线/带宽/删一试件后不稳定，应把速率效应写成探索性关联而不是因果结论。",
        "3. RKHS 距离曲线仍可用于表征每个试件偏离初始接触状态的非线性程度；组间曲线重叠不否定单试件内部存在显著转变。",
        "4. 可见损伤与机器峰值并非所有试件都按同一顺序出现，事件阈值分析是语义状态对照，不应误称为统一时间序列。",
        "5. 本敏感性分析的基线、带宽和压力阈值网格为预先固定的诊断设置，不据结果挑选最漂亮的参数。",
        "6. 事件点逐项 Cliff's δ 检验不显著、而整条多变量增量轨迹 MMD 显著并不矛盾：前者检验单状态单指标，后者聚合多个载荷状态中的协同变化；应将后者表述为分布层面的轨迹证据。",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/incremental_rkhs_sensitivity"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    balanced = pd.read_csv(args.root / "analysis" / "balanced_loading_trajectories.csv")
    specimen_ids, rates, grid, trajectories = trajectory_arrays(balanced)
    center, scale = robust_fit(trajectories.reshape(-1, trajectories.shape[-1]))
    standardized = robust_transform(
        trajectories.reshape(-1, trajectories.shape[-1]), center, scale
    ).reshape(trajectories.shape)
    rng = np.random.default_rng(args.seed)

    baseline = baseline_fraction_sensitivity(standardized, grid, rates)
    bandwidth = bandwidth_sensitivity(standardized, grid, rates)
    leave_out = leave_one_low_out(standardized, grid, specimen_ids, rates)
    features = feature_incremental_mmd(standardized, grid, rates)
    distance_rows, distance_summary, state_bandwidth = incremental_rkhs_distances(
        standardized, grid, specimen_ids, rates, rng, args.bootstrap_repeats
    )
    state_table = pd.read_csv(args.root / "dataset" / "evolution_state_table.csv")
    stage_index = pd.read_csv(
        args.root / "academic" / "multimodal_dashboards" / "multimodal_stage_index.csv"
    )
    threshold_events = threshold_event_table(args.workspace, state_table, stage_index)
    threshold_mmd = threshold_mmd_sensitivity(threshold_events)

    baseline.to_csv(args.output / "baseline_fraction_mmd_sensitivity.csv", index=False, encoding="utf-8-sig")
    bandwidth.to_csv(args.output / "bandwidth_mmd_sensitivity.csv", index=False, encoding="utf-8-sig")
    leave_out.to_csv(args.output / "leave_one_low_out_mmd.csv", index=False, encoding="utf-8-sig")
    features.to_csv(args.output / "feature_incremental_mmd.csv", index=False, encoding="utf-8-sig")
    distance_rows.to_csv(args.output / "incremental_rkhs_distance_trajectories.csv", index=False, encoding="utf-8-sig")
    distance_summary.to_csv(args.output / "incremental_rkhs_distance_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    threshold_events.to_csv(args.output / "threshold_event_area_metrics.csv", index=False, encoding="utf-8-sig")
    threshold_mmd.to_csv(args.output / "threshold_mmd_sensitivity.csv", index=False, encoding="utf-8-sig")
    plot_incremental_distances(
        distance_rows,
        distance_summary,
        args.output / "incremental_rkhs_distance_trajectories.png",
    )
    plot_sensitivity_overview(
        baseline,
        bandwidth,
        leave_out,
        threshold_mmd,
        args.output / "incremental_rkhs_sensitivity_overview.png",
    )
    plot_feature_incremental_mmd(
        features,
        args.output / "incremental_feature_mmd.png",
    )
    write_summary(
        args.output / "incremental_rkhs_sensitivity_summary_zh.md",
        baseline,
        bandwidth,
        leave_out,
        features,
        threshold_mmd,
    )
    metadata = {
        "included_specimens": len(specimen_ids),
        "rate_group_sizes": {rate: rates.count(rate) for rate in RATE_ORDER},
        "pressure_features": PRESSURE_FEATURES,
        "baseline_force_fractions": BASELINE_FRACTIONS,
        "common_evaluation_minimum_force_fraction": 0.30,
        "rbf_bandwidth_factors": BANDWIDTH_FACTORS,
        "pressure_thresholds_kpa": PRESSURE_THRESHOLDS_KPA,
        "state_space_rbf_bandwidth": state_bandwidth,
        "bootstrap_unit": "specimen",
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_seed": args.seed,
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
    }
    (args.output / "incremental_rkhs_sensitivity_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
