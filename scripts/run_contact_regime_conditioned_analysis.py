#!/usr/bin/env python3
"""Condition RKHS/load/damage analyses on rate-blind early contact regimes."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from run_early_contact_regime_analysis import EARLY_FEATURES, REGIME_COLORS
from run_event_stage_bootstrap_analysis import (
    RATE_COLORS,
    RATE_LABELS,
    benjamini_hochberg,
)
from run_incremental_rkhs_sensitivity import exact_mmd_test_fixed_bandwidth, trajectory_arrays
from run_rkhs_evolution_analysis import median_bandwidth, robust_fit, robust_transform


ASSOCIATIONS = {
    "transition_load_vs_damage_load": (
        "pressure_transition_force_fraction_of_peak",
        "damage_to_peak_force_ratio",
        "RKHS transition load / peak load",
        "Visible-damage load / peak load",
    ),
    "transition_lead_vs_damage_area": (
        "pressure_transition_lead_normalized_by_10pct_to_peak_duration",
        "maximum_visible_damage_area_fraction_pre_cutoff",
        "Transition lead / 10%-to-peak duration",
        "Maximum visible damage area fraction",
    ),
    "damage_state_distance_vs_damage_area": (
        "rkhs_distance_initial_to_damage_state",
        "maximum_visible_damage_area_fraction_pre_cutoff",
        "RKHS distance: initial to damage state",
        "Maximum visible damage area fraction",
    ),
    "transition_load_vs_peak_stress": (
        "pressure_transition_force_fraction_of_peak",
        "peak_nominal_stress_mpa",
        "RKHS transition load / peak load",
        "Peak nominal stress (MPa)",
    ),
}

PRESSURE_CHANGE_METRICS = {
    "pressure_area_ge_50kpa_fraction_of_face": "Area >=50 kPa / face",
    "pressure_effective_area_fraction_of_face": "Effective pressure area / face",
    "pressure_normalized_entropy": "Normalized pressure entropy",
    "pressure_top_10pct_cell_load_fraction": "Top-10% cell load fraction",
}

REGIME_NAMES = {1: "Broad/distributed", 2: "Localized/concentrated"}
RATE_MARKERS = {"low": "o", "medium": "s", "high": "^"}


def rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def spearman_rho(first: np.ndarray, second: np.ndarray) -> float:
    finite = np.isfinite(first) & np.isfinite(second)
    first_rank = rank_average(first[finite])
    second_rank = rank_average(second[finite])
    if len(first_rank) < 3 or np.std(first_rank) < 1e-12 or np.std(second_rank) < 1e-12:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def permutation_spearman(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float, int, str]:
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    first_rank = rank_average(first)
    second_rank = rank_average(second)
    first_centered = first_rank - np.mean(first_rank)
    second_centered = second_rank - np.mean(second_rank)
    denominator = float(np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2)))
    observed = (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    n = len(first)
    if not np.isfinite(observed):
        return observed, float("nan"), 0, "undefined"
    if n <= 9:
        exceed = 0
        total = 0
        for permutation in itertools.permutations(range(n)):
            statistic = float(np.dot(first_centered, second_centered[list(permutation)]) / denominator)
            exceed += abs(statistic) >= abs(observed) - 1e-12
            total += 1
        return observed, float(exceed / total), total, "exact"
    exceed = 0
    for _ in range(repeats):
        statistic = float(np.dot(first_centered, rng.permutation(second_centered)) / denominator)
        exceed += abs(statistic) >= abs(observed) - 1e-12
    return observed, float((exceed + 1) / (repeats + 1)), repeats, "monte_carlo"


def association_table(
    joined: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subsets = [("all", joined)] + [
        (REGIME_NAMES[index], joined[joined["regime_index_1_based"] == index]) for index in [1, 2]
    ]
    for subset_name, subset in subsets:
        for key, (x_name, y_name, x_label, y_label) in ASSOCIATIONS.items():
            rho, p_value, permutations, method = permutation_spearman(
                subset[x_name].to_numpy(dtype=float),
                subset[y_name].to_numpy(dtype=float),
                rng,
                repeats,
            )
            rows.append(
                {
                    "analysis_subset": subset_name,
                    "association": key,
                    "x_metric": x_name,
                    "y_metric": y_name,
                    "x_label": x_label,
                    "y_label": y_label,
                    "specimen_count": len(subset),
                    "spearman_rho": rho,
                    "permutation_p_value": p_value,
                    "permutation_count": permutations,
                    "permutation_method": method,
                }
            )
    table = pd.DataFrame(rows)
    table["bh_q_value_across_12_associations"] = benjamini_hochberg(
        table["permutation_p_value"].to_numpy(dtype=float)
    )
    return table


def event_pressure_changes(stage_table: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for specimen, rows in stage_table.groupby("specimen_id"):
        transition = rows[rows["stage"] == "rkhs_transition"].iloc[0]
        damage = rows[rows["stage"] == "visible_damage"].iloc[0]
        assignment = assignments[assignments["specimen_id"] == specimen].iloc[0]
        record: dict[str, object] = {
            "specimen_id": specimen,
            "rate_group": assignment["rate_group"],
            "regime_index_1_based": int(assignment["regime_index_1_based"]),
            "regime_name": assignment["regime_name"],
        }
        for metric in PRESSURE_CHANGE_METRICS:
            record[f"damage_minus_transition__{metric}"] = float(damage[metric]) - float(transition[metric])
        records.append(record)
    return pd.DataFrame(records)


def conditioned_mmd(
    balanced: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    specimen_ids, rates, grid, trajectories = trajectory_arrays(balanced)
    all_states = trajectories.reshape(-1, trajectories.shape[-1])
    center, scale = robust_fit(all_states)
    standardized = robust_transform(all_states, center, scale).reshape(trajectories.shape)
    assignment_by_id = assignments.set_index("specimen_id")
    localized = [
        index
        for index, specimen in enumerate(specimen_ids)
        if int(assignment_by_id.loc[specimen, "regime_index_1_based"]) == 2
    ]
    medium = [index for index in localized if rates[index] == "medium"]
    high = [index for index in localized if rates[index] == "high"]
    baseline_index = int(np.argmin(np.abs(grid - 0.40)))
    evaluation = np.flatnonzero(grid >= 0.50 - 1e-12)
    absolute = standardized[:, evaluation, :]
    adjusted = absolute - standardized[:, baseline_index : baseline_index + 1, :]
    rows: list[dict[str, object]] = []
    for name, values in [("absolute_50_to_100pct", absolute), ("increment_from_40pct", adjusted)]:
        joined = values[medium + high].reshape(len(medium) + len(high), -1)
        mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(joined, len(medium))
        rows.append(
            {
                "contact_regime": REGIME_NAMES[2],
                "comparison": "medium_vs_high",
                "trajectory_definition": name,
                "force_fraction_baseline": 0.40 if name.startswith("increment") else np.nan,
                "force_fraction_evaluation_minimum": 0.50,
                "specimen_count_medium": len(medium),
                "specimen_count_high": len(high),
                "mmd2": mmd2,
                "exact_permutation_p_value": p_value,
                "permutation_count": permutations,
                "rbf_bandwidth": bandwidth,
            }
        )
    feature_rows: list[dict[str, object]] = []
    for feature_index, feature in enumerate(EARLY_FEATURES):
        joined = adjusted[medium + high, :, feature_index]
        mmd2, p_value, permutations, bandwidth = exact_mmd_test_fixed_bandwidth(joined, len(medium))
        feature_rows.append(
            {
                "contact_regime": REGIME_NAMES[2],
                "comparison": "medium_vs_high",
                "pressure_feature": feature,
                "force_fraction_baseline": 0.40,
                "force_fraction_evaluation_minimum": 0.50,
                "specimen_count_medium": len(medium),
                "specimen_count_high": len(high),
                "incremental_feature_mmd2": mmd2,
                "exact_permutation_p_value": p_value,
                "permutation_count": permutations,
                "rbf_bandwidth": bandwidth,
            }
        )
    feature_table = pd.DataFrame(feature_rows)
    feature_table["bh_q_value_across_9_features"] = benjamini_hochberg(
        feature_table["exact_permutation_p_value"].to_numpy(dtype=float)
    )
    state_bandwidth = median_bandwidth(standardized.reshape(-1, standardized.shape[-1]))
    distance_rows: list[dict[str, object]] = []
    for specimen_index in localized:
        baseline = standardized[specimen_index, baseline_index]
        for force_index in np.flatnonzero(grid >= 0.40 - 1e-12):
            squared = float(np.sum((standardized[specimen_index, force_index] - baseline) ** 2))
            kernel_value = math.exp(-squared / (2.0 * state_bandwidth**2))
            distance_rows.append(
                {
                    "specimen_id": specimen_ids[specimen_index],
                    "rate_group": rates[specimen_index],
                    "force_fraction_of_specimen_peak": float(grid[force_index]),
                    "rkhs_distance_from_40pct_load_state": math.sqrt(max(0.0, 2.0 - 2.0 * kernel_value)),
                    "state_rbf_bandwidth": state_bandwidth,
                }
            )
    return pd.DataFrame(rows), feature_table, pd.DataFrame(distance_rows)


def plot_event_sequence(joined: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8), sharex=True)
    for axis, regime in zip(axes, [1, 2]):
        subset = joined[joined["regime_index_1_based"] == regime].copy()
        subset = subset.sort_values("pressure_transition_time_normalized_10pct_to_peak")
        y = np.arange(len(subset))
        for position, row in zip(y, subset.itertuples(index=False)):
            transition = row.pressure_transition_time_normalized_10pct_to_peak
            damage = row.damage_time_normalized_10pct_to_peak
            color = "#2563eb" if row.pressure_transition_precedes_damage else "#b91c1c"
            axis.plot([transition, damage], [position, position], color=color, lw=2.0, alpha=0.75)
            axis.scatter(transition, position, s=50, marker="o", color="#111827", zorder=3)
            axis.scatter(damage, position, s=58, marker="D", color=color, zorder=3)
        axis.axvline(1.0, color="#6b7280", lw=1.2, ls="--")
        axis.set_yticks(y)
        axis.set_yticklabels([value.replace("-5-rigid-", "") for value in subset["specimen_id"]])
        axis.set_title(f"{REGIME_NAMES[regime]} (n={len(subset)})")
        axis.set_xlabel("Normalized time from 10% load to machine peak")
        axis.grid(axis="x", alpha=0.18)
        axis.invert_yaxis()
    axes[0].set_ylabel("Specimen")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", label="RKHS transition"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#2563eb", label="Visible damage after transition"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#b91c1c", label="Visible damage before transition"),
        Line2D([0], [0], color="#6b7280", ls="--", label="Machine peak"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Event ordering within rate-blind early-contact regimes", fontsize=16, y=0.995)
    fig.text(0.5, 0.02, "Horizontal segments show chronology, not causality; times after machine peak exceed 1.0.", ha="center")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.13, wspace=0.28)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_pressure_changes(changes: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.8), sharey=True)
    rng = np.random.default_rng(20260927)
    metric_names = list(PRESSURE_CHANGE_METRICS)
    for axis, regime in zip(axes, [1, 2]):
        subset = changes[changes["regime_index_1_based"] == regime]
        for position, metric in enumerate(metric_names):
            values = subset[f"damage_minus_transition__{metric}"].to_numpy(dtype=float) * 100.0
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            axis.scatter(np.full(len(values), position) + jitter, values, s=38, alpha=0.72, color=REGIME_COLORS[regime - 1])
            q25, median, q75 = np.percentile(values, [25, 50, 75])
            axis.fill_betweenx(
                [q25, q75],
                position - 0.16,
                position + 0.16,
                color=REGIME_COLORS[regime - 1],
                alpha=0.18,
                zorder=1,
            )
            axis.plot([position - 0.14, position + 0.14], [median, median], color="#111827", lw=2.2, zorder=4)
        axis.axhline(0.0, color="#6b7280", lw=1.1)
        axis.set_xticks(range(len(metric_names)))
        axis.set_xticklabels([PRESSURE_CHANGE_METRICS[name] for name in metric_names], rotation=18, ha="right")
        axis.set_title(f"{REGIME_NAMES[regime]} (n={len(subset)})")
        axis.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("Visible-damage state minus RKHS-transition state (percentage points)")
    fig.suptitle("Pressure-topology redistribution between semantic event states", fontsize=16, y=0.995)
    fig.text(0.5, 0.02, "Dots are specimens; translucent boxes span the IQR; black ticks are medians. Four specimens have visible damage before the RKHS transition.", ha="center")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.24, wspace=0.16)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_associations(joined: pd.DataFrame, table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.0))
    for axis, (key, (x_name, y_name, x_label, y_label)) in zip(axes.flat, ASSOCIATIONS.items()):
        for regime in [1, 2]:
            subset = joined[joined["regime_index_1_based"] == regime]
            for row in subset.itertuples(index=False):
                axis.scatter(
                    getattr(row, x_name),
                    getattr(row, y_name),
                    marker=RATE_MARKERS[row.rate_group],
                    s=65,
                    facecolor=REGIME_COLORS[regime - 1],
                    edgecolor="#111827",
                    linewidth=0.7,
                    alpha=0.82,
                )
        annotations = []
        for regime in [1, 2]:
            result = table[(table["analysis_subset"] == REGIME_NAMES[regime]) & (table["association"] == key)].iloc[0]
            annotations.append(
                f"R{regime}: rho={result.spearman_rho:.2f}, q={result.bh_q_value_across_12_associations:.3f}"
            )
        axis.text(
            0.02,
            0.98,
            "\n".join(annotations),
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.0},
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.18)
    regime_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[index - 1], markeredgecolor="#111827", label=f"R{index}: {REGIME_NAMES[index]}")
        for index in [1, 2]
    ]
    rate_handles = [
        Line2D([0], [0], marker=RATE_MARKERS[rate], color="#111827", linestyle="none", markerfacecolor="white", label=RATE_LABELS[rate])
        for rate in ["low", "medium", "high"]
    ]
    fig.legend(handles=regime_handles + rate_handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("RKHS transition, load and visible-damage associations within contact regimes", fontsize=16, y=0.995)
    fig.text(0.5, 0.02, "rho: Spearman rank correlation; q: BH adjustment across 12 exploratory associations. No fitted line is shown for these small groups.", ha="center")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.08, hspace=0.30, wspace=0.24)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_localized_incremental(
    state_distances: pd.DataFrame,
    mmd_table: pd.DataFrame,
    output: Path,
) -> None:
    table = state_distances.copy()
    fig, axis = plt.subplots(figsize=(11.5, 7.0))
    for rate in ["medium", "high"]:
        subset = table[table["rate_group"] == rate]
        for _, rows in subset.groupby("specimen_id"):
            rows = rows.sort_values("force_fraction_of_specimen_peak")
            axis.plot(
                rows["force_fraction_of_specimen_peak"],
                rows["rkhs_distance_from_40pct_load_state"],
                color=RATE_COLORS[rate],
                lw=0.9,
                alpha=0.25,
            )
        grouped = subset.groupby("force_fraction_of_specimen_peak")["rkhs_distance_from_40pct_load_state"]
        x, medians, q25s, q75s = [], [], [], []
        for force_fraction, values in grouped:
            x.append(float(force_fraction))
            q25, median, q75 = np.percentile(values.to_numpy(dtype=float), [25, 50, 75])
            medians.append(median)
            q25s.append(q25)
            q75s.append(q75)
        axis.fill_between(x, q25s, q75s, color=RATE_COLORS[rate], alpha=0.14)
        axis.plot(x, medians, color=RATE_COLORS[rate], lw=2.5, marker="o", ms=4, label=f"{rate.title()} (n={subset['specimen_id'].nunique()})")
    adjusted = mmd_table[mmd_table["trajectory_definition"] == "increment_from_40pct"].iloc[0]
    axis.axvline(0.40, color="#6b7280", lw=1.1, ls="--")
    axis.text(
        0.02,
        0.97,
        f"Post-40% adjusted trajectory MMD²={adjusted.mmd2:.3f}, exact p={adjusted.exact_permutation_p_value:.4f}",
        transform=axis.transAxes,
        fontsize=10,
        va="top",
    )
    axis.set_xlabel("Machine force / specimen peak force")
    axis.set_ylabel("RKHS distance from the specimen's 40%-load pressure state")
    axis.set_title("Localized-contact regime: later RKHS evolution by loading-rate group")
    axis.grid(alpha=0.20)
    axis.legend(frameon=False)
    fig.text(0.5, 0.02, "Thin lines are specimens; thick lines are medians; bands are IQRs. The exact MMD test uses 50%-100% states after subtracting each specimen's 40% state.", ha="center")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.14)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/contact_regime_conditioned"),
    )
    parser.add_argument("--permutation-repeats", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    academic = args.root / "academic"
    assignments = pd.read_csv(academic / "early_contact_regimes" / "early_contact_regime_assignments.csv")
    metrics = pd.read_csv(academic / "ice_load_specimen_metrics.csv")
    joined = assignments.merge(metrics, on=["specimen_id", "rate_group"], validate="one_to_one")
    rng = np.random.default_rng(args.seed)
    associations = association_table(joined, rng, args.permutation_repeats)
    stage_table = pd.read_csv(academic / "bootstrap_event_analysis" / "event_stage_pressure_metrics.csv")
    changes = event_pressure_changes(stage_table, assignments)
    balanced = pd.read_csv(args.root / "analysis" / "balanced_loading_trajectories.csv")
    mmd_table, feature_mmd, state_distances = conditioned_mmd(balanced, assignments)

    joined.to_csv(args.output / "conditional_event_sequence_metrics.csv", index=False, encoding="utf-8-sig")
    associations.to_csv(args.output / "within_regime_associations.csv", index=False, encoding="utf-8-sig")
    changes.to_csv(args.output / "transition_to_damage_pressure_changes.csv", index=False, encoding="utf-8-sig")
    mmd_table.to_csv(args.output / "localized_rate_conditioned_mmd.csv", index=False, encoding="utf-8-sig")
    feature_mmd.to_csv(args.output / "localized_rate_conditioned_feature_mmd.csv", index=False, encoding="utf-8-sig")
    state_distances.to_csv(args.output / "localized_post40_rkhs_distances.csv", index=False, encoding="utf-8-sig")

    plot_event_sequence(joined, args.output / "conditional_event_sequence.png")
    plot_pressure_changes(changes, args.output / "conditional_pressure_state_changes_iqr.png")
    plot_associations(joined, associations, args.output / "conditional_rkhs_damage_associations.png")
    plot_localized_incremental(
        state_distances,
        mmd_table,
        args.output / "localized_rate_conditioned_rkhs_iqr.png",
    )

    summary_lines = [
        "# 接触机制条件化的 RKHS—载荷—可见破坏分析",
        "",
        "- 接触机制标签来自不使用速率、载荷和破坏标签的早期压力拓扑聚类。",
        "- 宽域/分布式机制仅 n=4，任何组内相关都只作描述；精确双侧置换检验的最小分辨率受 4!=24 限制。",
        "- 局部/集中式机制 n=12，可检验其中 4 个中速与 8 个高速试件在相同早期接触机制下的后续压力演化差异。",
        "- 事件状态差使用‘可见损伤状态减 RKHS 转变状态’，但 4 个试件中可见损伤在时间上早于 RKHS 转变，因此不能把差值统一解释为因果增量。",
        "",
        "## 组内相关",
        "",
    ]
    for row in associations.itertuples(index=False):
        summary_lines.append(
            f"- {row.analysis_subset}｜{row.association}: n={row.specimen_count}, rho={row.spearman_rho:.3f}, "
            f"p={row.permutation_p_value:.5f}, q={row.bh_q_value_across_12_associations:.5f}."
        )
    summary_lines.extend(["", "## 同一局部接触机制内的中速—高速比较", ""])
    for row in mmd_table.itertuples(index=False):
        summary_lines.append(
            f"- {row.trajectory_definition}: MMD²={row.mmd2:.4f}, exact p={row.exact_permutation_p_value:.5f} "
            f"({row.permutation_count} 个标签组合)."
        )
    significant = feature_mmd[feature_mmd["bh_q_value_across_9_features"] <= 0.05]
    summary_lines.append(
        f"- 40% 状态校正后的 9 个单特征轨迹中，BH q<=0.05 的特征数为 {len(significant)}。"
    )
    summary_lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "1. 这是条件化的探索性关联分析，不是接触机制的随机分层试验。",
            "2. 接触机制由早期 10%–40% 压力状态定义；后续速率比较只使用 50%–100% 状态，并另做 40% 基线相减，以降低循环论证。",
            "3. 图中保留全部试件点或轨迹，并用中位数/IQR描述分布；完整 p/q 与效应统计保存在 CSV 中。",
            "4. 未经显著性和留出验证支持，不得把组内趋势写成冰载荷预测规律。",
        ]
    )
    (args.output / "contact_regime_conditioned_summary_zh.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    metadata = {
        "included_specimens": len(joined),
        "regime_counts": joined["regime_name"].value_counts().to_dict(),
        "association_permutation_repeats": args.permutation_repeats,
        "localized_rate_comparison": {"medium": 4, "high": 8},
        "early_regime_definition_force_range": [0.10, 0.40],
        "conditioned_evolution_evaluation_force_range": [0.50, 1.00],
        "conditioned_evolution_baseline_force_fraction": 0.40,
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
        "random_seed": args.seed,
    }
    (args.output / "contact_regime_conditioned_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
