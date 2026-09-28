#!/usr/bin/env python3
"""Specimen-level uncertainty and event-pair spatial statistics for RKHS ice tests."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATE_ORDER = ["low", "medium", "high"]
RATE_LABELS = {"low": "Low (n=3)", "medium": "Medium (n=5)", "high": "High (n=8)"}
RATE_COLORS = {"low": "#2563eb", "medium": "#d97706", "high": "#dc2626"}
STAGE_ORDER = ["pre_transition", "rkhs_transition", "visible_damage", "machine_peak"]
STAGE_LABELS = {
    "pre_transition": "Pre-transition",
    "rkhs_transition": "RKHS transition",
    "visible_damage": "Visible damage",
    "machine_peak": "Machine peak",
}
PAIR_ORDER = [
    "pre_transition__rkhs_transition",
    "rkhs_transition__visible_damage",
    "visible_damage__machine_peak",
    "pre_transition__visible_damage",
]
PAIR_LABELS = {
    "pre_transition__rkhs_transition": "Pre ↔ RKHS",
    "rkhs_transition__visible_damage": "RKHS ↔ damage",
    "visible_damage__machine_peak": "Damage ↔ peak",
    "pre_transition__visible_damage": "Pre ↔ damage",
}
PRIMARY_METRICS = {
    "pressure_area_ge_50kpa_fraction_of_face": ("Area ≥50 kPa / face", "%", 100.0),
    "pressure_effective_area_fraction_of_face": ("Effective pressure area / face", "%", 100.0),
    "pressure_normalized_entropy": ("Normalized pressure entropy", "–", 1.0),
    "pressure_top_10pct_cell_load_fraction": ("Top-10% cell load fraction", "%", 100.0),
}
DELTA_METRICS = {
    f"delta_{metric}": (f"Change in {label[0].lower() + label[1:]}", unit, scale)
    for metric, (label, unit, scale) in PRIMARY_METRICS.items()
}
SPATIAL_METRICS = {
    "hellinger_distance": ("Hellinger pressure-field distance", "–", 1.0),
    "total_variation_distance": ("Total-variation distance", "–", 1.0),
    "centroid_displacement_mm": ("Load-centroid displacement", "mm", 1.0),
    "support_iou_ge_50kpa": ("≥50 kPa support overlap (IoU)", "–", 1.0),
}


def stage_key(label: str) -> str:
    if label.startswith("Pre-transition"):
        return "pre_transition"
    if label.startswith("RKHS transition"):
        return "rkhs_transition"
    if label == "Sustained visible damage":
        return "visible_damage"
    if label == "Machine peak load":
        return "machine_peak"
    raise ValueError(f"Unknown stage label: {label}")


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def bootstrap_median(values: np.ndarray, rng: np.random.Generator, repeats: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.full(repeats, np.nan)
    draws = values[rng.integers(0, values.size, size=(repeats, values.size))]
    return np.median(draws, axis=1)


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if first.size == 0 or second.size == 0:
        return float("nan")
    differences = second[:, None] - first[None, :]
    return float((np.sum(differences > 0) - np.sum(differences < 0)) / differences.size)


def bootstrap_two_group(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    repeats: int,
) -> tuple[np.ndarray, np.ndarray]:
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    first_draws = first[rng.integers(0, first.size, size=(repeats, first.size))]
    second_draws = second[rng.integers(0, second.size, size=(repeats, second.size))]
    median_differences = np.median(second_draws, axis=1) - np.median(first_draws, axis=1)
    delta_draws = np.empty(repeats, dtype=float)
    for index in range(repeats):
        delta_draws[index] = cliffs_delta(first_draws[index], second_draws[index])
    return median_differences, delta_draws


def exact_permutation_p_median(first: np.ndarray, second: np.ndarray) -> float:
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    pooled = np.concatenate([first, second])
    n_first = first.size
    observed = abs(float(np.median(second) - np.median(first)))
    exceed = 0
    total = 0
    all_indices = np.arange(pooled.size)
    for chosen in itertools.combinations(range(pooled.size), n_first):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(chosen)] = True
        statistic = abs(float(np.median(pooled[~mask]) - np.median(pooled[mask])))
        exceed += statistic >= observed - 1e-12
        total += 1
    return float(exceed / total)


def exact_permutation_p_cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    pooled = np.concatenate([first, second])
    n_first = first.size
    observed = abs(cliffs_delta(first, second))
    exceed = 0
    total = 0
    for chosen in itertools.combinations(range(pooled.size), n_first):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(chosen)] = True
        statistic = abs(cliffs_delta(pooled[mask], pooled[~mask]))
        exceed += statistic >= observed - 1e-12
        total += 1
    return float(exceed / total)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    result = np.full_like(p_values, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return result
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.minimum(ranked, 1.0)
    return result


def pressure_distribution(field: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    weighted = np.maximum(np.asarray(field, dtype=float), 0.0) * np.asarray(overlap, dtype=float)
    total = float(np.sum(weighted))
    if total <= 0:
        return np.full(weighted.size, 1.0 / weighted.size)
    return (weighted / total).ravel()


def pressure_centroid_mm(distribution: np.ndarray) -> tuple[float, float]:
    coordinates = np.linspace(-35.0 + 35.0 / 32.0, 35.0 - 35.0 / 32.0, 32)
    xx, yy = np.meshgrid(coordinates, coordinates)
    field = distribution.reshape(32, 32)
    return float(np.sum(field * xx)), float(np.sum(field * yy))


def weighted_support_iou(
    first: np.ndarray,
    second: np.ndarray,
    overlap: np.ndarray,
    threshold_kpa: float = 50.0,
) -> float:
    first_support = (np.asarray(first) >= threshold_kpa) * overlap
    second_support = (np.asarray(second) >= threshold_kpa) * overlap
    union = float(np.sum(np.maximum(first_support, second_support)))
    if union <= 0:
        return 1.0
    return float(np.sum(np.minimum(first_support, second_support)) / union)


def build_stage_table(
    workspace: Path,
    state_table: pd.DataFrame,
    stage_index: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    fields: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for record in stage_index.to_dict("records"):
        specimen = str(record["specimen_id"])
        key = stage_key(str(record["stage"]))
        candidates = state_table[state_table["specimen_id"] == specimen]
        exact = candidates[candidates["pressure_frame_number"] == int(record["pressure_frame_number"])]
        if exact.empty:
            nearest_index = (candidates["time_s"] - float(record["pressure_frame_time_s"])).abs().idxmin()
            state = candidates.loc[nearest_index]
        else:
            state = exact.iloc[0]
        tensor_path = workspace / "data_interim" / specimen / "unified" / "pressure_fields_32x32.npz"
        with np.load(tensor_path) as tensor:
            tensor_index = int(state["pressure_field_tensor_index_0_based"])
            field = np.asarray(tensor["pressure_kpa"][tensor_index], dtype=float)
            overlap = np.asarray(tensor["footprint_overlap_fraction"], dtype=float)
        fields[(specimen, key)] = (field.copy(), overlap.copy())
        row = {
            "specimen_id": specimen,
            "rate_group": str(record["rate_group"]),
            "stage": key,
            "stage_label": STAGE_LABELS[key],
            "machine_time_s": float(record["machine_time_s"]),
            "pressure_frame_time_s": float(record["pressure_frame_time_s"]),
            "pressure_frame_number": int(record["pressure_frame_number"]),
            "force_fraction_of_specimen_peak": float(record["force_fraction_of_specimen_peak"]),
        }
        for metric in PRIMARY_METRICS:
            row[metric] = float(state[metric])
        rows.append(row)
    table = pd.DataFrame(rows)
    table["stage"] = pd.Categorical(table["stage"], STAGE_ORDER, ordered=True)
    return table.sort_values(["rate_group", "specimen_id", "stage"]).reset_index(drop=True), fields


def build_spatial_pair_table(
    stage_table: pd.DataFrame,
    fields: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rate_by_specimen = stage_table.groupby("specimen_id", observed=True)["rate_group"].first()
    time_by_key = {
        (str(row.specimen_id), str(row.stage)): float(row.machine_time_s)
        for row in stage_table.itertuples(index=False)
    }
    for specimen, rate in rate_by_specimen.items():
        for pair in PAIR_ORDER:
            first_key, second_key = pair.split("__")
            first_field, overlap = fields[(specimen, first_key)]
            second_field, _ = fields[(specimen, second_key)]
            first_distribution = pressure_distribution(first_field, overlap)
            second_distribution = pressure_distribution(second_field, overlap)
            hellinger = np.linalg.norm(np.sqrt(first_distribution) - np.sqrt(second_distribution)) / np.sqrt(2.0)
            total_variation = 0.5 * float(np.sum(np.abs(first_distribution - second_distribution)))
            first_centroid = pressure_centroid_mm(first_distribution)
            second_centroid = pressure_centroid_mm(second_distribution)
            centroid_displacement = float(np.hypot(
                second_centroid[0] - first_centroid[0],
                second_centroid[1] - first_centroid[1],
            ))
            time_difference = time_by_key[(specimen, second_key)] - time_by_key[(specimen, first_key)]
            rows.append(
                {
                    "specimen_id": specimen,
                    "rate_group": rate,
                    "event_pair": pair,
                    "event_pair_label": PAIR_LABELS[pair],
                    "signed_machine_time_difference_s": time_difference,
                    "listed_order_is_chronological": int(time_difference >= 0.0),
                    "hellinger_distance": float(hellinger),
                    "total_variation_distance": total_variation,
                    "centroid_displacement_mm": centroid_displacement,
                    "support_iou_ge_50kpa": weighted_support_iou(first_field, second_field, overlap),
                }
            )
    table = pd.DataFrame(rows)
    table["event_pair"] = pd.Categorical(table["event_pair"], PAIR_ORDER, ordered=True)
    return table.sort_values(["rate_group", "specimen_id", "event_pair"]).reset_index(drop=True)


def build_baseline_change_table(stage_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for specimen, specimen_rows in stage_table.groupby("specimen_id", observed=True):
        baseline = specimen_rows[specimen_rows["stage"].astype(str) == "pre_transition"].iloc[0]
        for stage in STAGE_ORDER:
            current = specimen_rows[specimen_rows["stage"].astype(str) == stage].iloc[0]
            row: dict[str, object] = {
                "specimen_id": specimen,
                "rate_group": current["rate_group"],
                "stage": stage,
                "stage_label": STAGE_LABELS[stage],
            }
            for metric in PRIMARY_METRICS:
                row[f"delta_{metric}"] = float(current[metric]) - float(baseline[metric])
            rows.append(row)
    result = pd.DataFrame(rows)
    result["stage"] = pd.Categorical(result["stage"], STAGE_ORDER, ordered=True)
    return result.sort_values(["rate_group", "specimen_id", "stage"]).reset_index(drop=True)


def summarize_bootstrap(
    table: pd.DataFrame,
    group_column: str,
    group_order: list[str],
    metrics: dict[str, tuple[str, str, float]],
    rng: np.random.Generator,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rate in RATE_ORDER:
        for group in group_order:
            subset = table[(table["rate_group"] == rate) & (table[group_column].astype(str) == group)]
            for metric in metrics:
                values = subset[metric].to_numpy(dtype=float)
                samples = bootstrap_median(values, rng, repeats)
                ci_low, ci_high = percentile_interval(samples)
                rows.append(
                    {
                        "rate_group": rate,
                        group_column: group,
                        "metric": metric,
                        "n_specimens": int(np.isfinite(values).sum()),
                        "median": float(np.nanmedian(values)),
                        "bootstrap_ci95_low": ci_low,
                        "bootstrap_ci95_high": ci_high,
                        "bootstrap_repeats": repeats,
                    }
                )
    return pd.DataFrame(rows)


def low_high_effects(
    table: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
    stages: list[str],
    metrics: dict[str, tuple[str, str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stage in stages:
        subset = table[table["stage"].astype(str) == stage]
        for metric in metrics:
            low = subset[subset["rate_group"] == "low"][metric].to_numpy(dtype=float)
            high = subset[subset["rate_group"] == "high"][metric].to_numpy(dtype=float)
            difference_draws, delta_draws = bootstrap_two_group(low, high, rng, repeats)
            diff_low, diff_high = percentile_interval(difference_draws)
            delta_low, delta_high = percentile_interval(delta_draws)
            rows.append(
                {
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "metric": metric,
                    "comparison": "high_minus_low",
                    "n_low": int(np.isfinite(low).sum()),
                    "n_high": int(np.isfinite(high).sum()),
                    "median_difference_high_minus_low": float(np.nanmedian(high) - np.nanmedian(low)),
                    "median_difference_bootstrap_ci95_low": diff_low,
                    "median_difference_bootstrap_ci95_high": diff_high,
                    "cliffs_delta_high_vs_low": cliffs_delta(low, high),
                    "cliffs_delta_bootstrap_ci95_low": delta_low,
                    "cliffs_delta_bootstrap_ci95_high": delta_high,
                    "exact_permutation_p_cliffs_delta": exact_permutation_p_cliffs_delta(low, high),
                    "exact_permutation_p_median_difference": exact_permutation_p_median(low, high),
                }
            )
    result = pd.DataFrame(rows)
    result["bh_q_cliffs_delta"] = benjamini_hochberg(
        result["exact_permutation_p_cliffs_delta"].to_numpy(dtype=float)
    )
    result["bh_q_median_difference"] = benjamini_hochberg(
        result["exact_permutation_p_median_difference"].to_numpy(dtype=float)
    )
    result["multiplicity_family_size"] = len(result)
    return result


def plot_stage_metrics(
    stage_table: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    positions = np.arange(len(STAGE_ORDER), dtype=float)
    offsets = {"low": -0.18, "medium": 0.0, "high": 0.18}
    rng = np.random.default_rng(20260928)
    for axis, (metric, (label, unit, scale)) in zip(axes.flat, PRIMARY_METRICS.items()):
        for rate in RATE_ORDER:
            subset = stage_table[stage_table["rate_group"] == rate]
            rate_summary = summary[(summary["rate_group"] == rate) & (summary["metric"] == metric)]
            medians = []
            lows = []
            highs = []
            for stage in STAGE_ORDER:
                row = rate_summary[rate_summary["stage"] == stage].iloc[0]
                medians.append(float(row["median"]) * scale)
                lows.append(float(row["bootstrap_ci95_low"]) * scale)
                highs.append(float(row["bootstrap_ci95_high"]) * scale)
                values = subset[subset["stage"].astype(str) == stage][metric].to_numpy(dtype=float) * scale
                jitter = rng.uniform(-0.035, 0.035, size=values.size)
                axis.scatter(
                    np.full(values.size, positions[STAGE_ORDER.index(stage)] + offsets[rate]) + jitter,
                    values,
                    s=22,
                    color=RATE_COLORS[rate],
                    alpha=0.28,
                    edgecolors="none",
                    zorder=1,
                )
            x = positions + offsets[rate]
            median_array = np.asarray(medians)
            axis.plot(x, median_array, color=RATE_COLORS[rate], lw=1.5, alpha=0.8)
            axis.errorbar(
                x,
                median_array,
                yerr=[median_array - np.asarray(lows), np.asarray(highs) - median_array],
                fmt="o",
                ms=6,
                capsize=3,
                lw=1.4,
                color=RATE_COLORS[rate],
                label=RATE_LABELS[rate],
                zorder=3,
            )
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(positions, [STAGE_LABELS[stage] for stage in STAGE_ORDER], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Specimen-level pressure-state medians with 95% bootstrap intervals", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.900,
        "Resampling unit: specimen. Event-defined stages are not necessarily chronological.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.855, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_spatial_pairs(
    spatial_table: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    positions = np.arange(len(PAIR_ORDER), dtype=float)
    offsets = {"low": -0.18, "medium": 0.0, "high": 0.18}
    rng = np.random.default_rng(20260929)
    for axis, (metric, (label, unit, scale)) in zip(axes.flat, SPATIAL_METRICS.items()):
        for rate in RATE_ORDER:
            subset = spatial_table[spatial_table["rate_group"] == rate]
            rate_summary = summary[(summary["rate_group"] == rate) & (summary["metric"] == metric)]
            medians = []
            lows = []
            highs = []
            for pair in PAIR_ORDER:
                row = rate_summary[rate_summary["event_pair"] == pair].iloc[0]
                medians.append(float(row["median"]) * scale)
                lows.append(float(row["bootstrap_ci95_low"]) * scale)
                highs.append(float(row["bootstrap_ci95_high"]) * scale)
                values = subset[subset["event_pair"].astype(str) == pair][metric].to_numpy(dtype=float) * scale
                jitter = rng.uniform(-0.035, 0.035, size=values.size)
                axis.scatter(
                    np.full(values.size, positions[PAIR_ORDER.index(pair)] + offsets[rate]) + jitter,
                    values,
                    s=22,
                    color=RATE_COLORS[rate],
                    alpha=0.28,
                    edgecolors="none",
                    zorder=1,
                )
            x = positions + offsets[rate]
            medians_array = np.asarray(medians)
            axis.plot(x, medians_array, color=RATE_COLORS[rate], lw=1.5, alpha=0.8)
            axis.errorbar(
                x,
                medians_array,
                yerr=[medians_array - np.asarray(lows), np.asarray(highs) - medians_array],
                fmt="o",
                ms=6,
                capsize=3,
                lw=1.4,
                color=RATE_COLORS[rate],
                label=RATE_LABELS[rate],
                zorder=3,
            )
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(positions, [PAIR_LABELS[pair] for pair in PAIR_ORDER], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Spatial redistribution between event-defined pressure fields", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.900,
        "Distances are symmetric; arrows do not imply temporal order. Whiskers are specimen-bootstrap 95% intervals.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.855, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_baseline_changes(
    change_table: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    stages = STAGE_ORDER[1:]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    positions = np.arange(len(stages), dtype=float)
    offsets = {"low": -0.18, "medium": 0.0, "high": 0.18}
    rng = np.random.default_rng(20260930)
    for axis, (metric, (label, unit, scale)) in zip(axes.flat, DELTA_METRICS.items()):
        for rate in RATE_ORDER:
            subset = change_table[change_table["rate_group"] == rate]
            rate_summary = summary[(summary["rate_group"] == rate) & (summary["metric"] == metric)]
            medians = []
            lows = []
            highs = []
            for stage in stages:
                row = rate_summary[rate_summary["stage"] == stage].iloc[0]
                medians.append(float(row["median"]) * scale)
                lows.append(float(row["bootstrap_ci95_low"]) * scale)
                highs.append(float(row["bootstrap_ci95_high"]) * scale)
                values = subset[subset["stage"].astype(str) == stage][metric].to_numpy(dtype=float) * scale
                jitter = rng.uniform(-0.035, 0.035, size=values.size)
                axis.scatter(
                    np.full(values.size, positions[stages.index(stage)] + offsets[rate]) + jitter,
                    values,
                    s=22,
                    color=RATE_COLORS[rate],
                    alpha=0.28,
                    edgecolors="none",
                    zorder=1,
                )
            x = positions + offsets[rate]
            median_array = np.asarray(medians)
            axis.plot(x, median_array, color=RATE_COLORS[rate], lw=1.5, alpha=0.8)
            axis.errorbar(
                x,
                median_array,
                yerr=[median_array - np.asarray(lows), np.asarray(highs) - median_array],
                fmt="o",
                ms=6,
                capsize=3,
                lw=1.4,
                color=RATE_COLORS[rate],
                label=RATE_LABELS[rate],
                zorder=3,
            )
        axis.axhline(0.0, color="#4b5563", lw=1.0)
        axis.set_title(label)
        axis.set_ylabel(unit)
        axis.set_xticks(positions, [STAGE_LABELS[stage] for stage in stages], rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Within-specimen pressure evolution relative to the pre-transition state", fontsize=16, y=0.995)
    fig.text(
        0.5,
        0.900,
        "Baseline subtraction removes persistent between-specimen contact offsets; whiskers are specimen-bootstrap 95% intervals.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.855, hspace=0.32, wspace=0.22)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_effect_sizes(effects: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    y = np.arange(len(STAGE_ORDER))
    for axis, (metric, (label, _, _)) in zip(axes.flat, PRIMARY_METRICS.items()):
        subset = effects[effects["metric"] == metric].copy()
        subset["stage"] = pd.Categorical(subset["stage"], STAGE_ORDER, ordered=True)
        subset = subset.sort_values("stage")
        point = subset["cliffs_delta_high_vs_low"].to_numpy(dtype=float)
        lower = subset["cliffs_delta_bootstrap_ci95_low"].to_numpy(dtype=float)
        upper = subset["cliffs_delta_bootstrap_ci95_high"].to_numpy(dtype=float)
        axis.axvline(0.0, color="#4b5563", lw=1.0)
        axis.axvspan(-0.147, 0.147, color="#9ca3af", alpha=0.12)
        axis.errorbar(
            point,
            y,
            xerr=[point - lower, upper - point],
            fmt="o",
            color="#7c3aed",
            capsize=3,
            lw=1.5,
        )
        for row_index, (_, row) in enumerate(subset.iterrows()):
            label_x = -1.00 if float(row["cliffs_delta_high_vs_low"]) > 0.60 else 1.25
            label_alignment = "left" if label_x < 0 else "right"
            axis.text(
                label_x,
                row_index,
                f"q={float(row['bh_q_cliffs_delta']):.3f}",
                va="center",
                ha=label_alignment,
                fontsize=9,
            )
        axis.set_title(label)
        axis.set_yticks(y, [STAGE_LABELS[stage] for stage in STAGE_ORDER])
        axis.set_xlim(-1.05, 1.30)
        axis.set_xlabel("Cliff's δ (positive: high rate > low rate)")
        axis.grid(axis="x", alpha=0.22)
    fig.suptitle("Low–high rate effect sizes at matched event states", fontsize=16, y=0.99)
    fig.text(
        0.5,
        0.945,
        "Points: Cliff's δ; whiskers: specimen-bootstrap 95% intervals; q: BH-adjusted exact permutation test of Cliff's δ",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.08, top=0.90, hspace=0.30, wspace=0.34)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_summary(
    output: Path,
    stage_table: pd.DataFrame,
    spatial_table: pd.DataFrame,
    effects: pd.DataFrame,
    baseline_effects: pd.DataFrame,
    repeats: int,
) -> None:
    significant = effects[effects["bh_q_cliffs_delta"] <= 0.05]
    significant_baseline = baseline_effects[baseline_effects["bh_q_cliffs_delta"] <= 0.05]
    reverse_transition_damage = int(
        spatial_table[
            spatial_table["event_pair"].astype(str) == "rkhs_transition__visible_damage"
        ]["listed_order_is_chronological"].eq(0).sum()
    )
    reverse_damage_peak = int(
        spatial_table[
            spatial_table["event_pair"].astype(str) == "visible_damage__machine_peak"
        ]["listed_order_is_chronological"].eq(0).sum()
    )
    lines = [
        "# 事件分阶段压力统计与试件级 Bootstrap 摘要",
        "",
        f"- 纳入 {stage_table['specimen_id'].nunique()} 个独立试件：低速 3、中速 5、高速 8；210-7、210-9 继续排除。",
        f"- Bootstrap 重采样单位是试件而不是帧，共 {repeats:,} 次；区间为百分位 95% 区间。",
        "- 低速组仅 3 个试件，置信区间呈离散且对单个试件敏感，应报告效应量与原始散点，不应只依据显著性。",
        "- 四个状态是事件定义的对照状态，不保证时间顺序；空间距离均按对称距离计算。",
        f"- 可见损伤早于 RKHS 转变的试件为 {reverse_transition_damage}/16；机器峰值早于可见损伤的试件为 {reverse_damage_peak}/16。",
        f"- 16 个低速—高速『状态×指标』Cliff's δ 精确置换检验经 BH 校正后 q≤0.05 的数量为 {len(significant)}。",
        f"- 对每个试件先减去自身转变前基线后，12 个『后续状态×指标』低速—高速检验中 q≤0.05 的数量为 {len(significant_baseline)}。",
        "",
        "## 统计解释边界",
        "",
        "1. Bootstrap 区间描述当前试件集合对组中位数的不确定性，不等同于大样本总体置信保证。",
        "2. Cliff's δ 为秩效应量：正值表示高速组更大，负值表示低速组更大；其区间跨 0 时不应宣称方向已稳定。",
        "3. 图中 q 值来自 Cliff's δ 的精确置换检验，并对 16 个状态—指标检验统一进行 Benjamini–Hochberg 校正；表中同时保留中位数差检验供审计。",
        "4. 当当前样本实现完全组间分离时，普通百分位 Bootstrap 的 Cliff's δ 区间会退化到 +1 或 -1；这只是条件于已观测试件的经验分离，不能证明总体无重叠。",
        "5. 压力单元大量饱和，因此主结论优先使用接触面积、有效面积、熵、载荷集中度和归一化空间距离，不依赖绝对峰值压力。",
        "6. 事件对距离揭示压力场重排幅度，但不能单独证明其导致了侧面裂纹或破碎。",
        "7. 绝对状态的组间分离若在转变前已经存在，而基线校正后的变化不再分离，应优先解释为试件表面、摆放或批次条件形成的接触状态差异，而不是加载速率改变了演化路径。",
    ]
    if len(significant):
        lines.extend(["", "## BH 校正后仍显著的低速—高速对比", ""])
        for row in significant.itertuples(index=False):
            label = PRIMARY_METRICS[row.metric][0]
            lines.append(
                f"- {STAGE_LABELS[row.stage]}，{label}：Cliff's δ={row.cliffs_delta_high_vs_low:.3f}，"
                f"q={row.bh_q_cliffs_delta:.4f}。"
            )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/bootstrap_event_analysis"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    state_table = pd.read_csv(args.root / "dataset" / "evolution_state_table.csv")
    stage_index = pd.read_csv(
        args.root / "academic" / "multimodal_dashboards" / "multimodal_stage_index.csv"
    )
    stage_table, fields = build_stage_table(args.workspace, state_table, stage_index)
    spatial_table = build_spatial_pair_table(stage_table, fields)
    change_table = build_baseline_change_table(stage_table)
    rng = np.random.default_rng(args.seed)
    stage_summary = summarize_bootstrap(
        stage_table,
        "stage",
        STAGE_ORDER,
        PRIMARY_METRICS,
        rng,
        args.bootstrap_repeats,
    )
    spatial_summary = summarize_bootstrap(
        spatial_table,
        "event_pair",
        PAIR_ORDER,
        SPATIAL_METRICS,
        rng,
        args.bootstrap_repeats,
    )
    change_summary = summarize_bootstrap(
        change_table,
        "stage",
        STAGE_ORDER,
        DELTA_METRICS,
        rng,
        args.bootstrap_repeats,
    )
    effects = low_high_effects(
        stage_table,
        rng,
        args.bootstrap_repeats,
        STAGE_ORDER,
        PRIMARY_METRICS,
    )
    baseline_effects = low_high_effects(
        change_table,
        rng,
        args.bootstrap_repeats,
        STAGE_ORDER[1:],
        DELTA_METRICS,
    )

    stage_table.to_csv(args.output / "event_stage_pressure_metrics.csv", index=False, encoding="utf-8-sig")
    spatial_table.to_csv(args.output / "event_pair_spatial_statistics.csv", index=False, encoding="utf-8-sig")
    change_table.to_csv(args.output / "baseline_adjusted_event_changes.csv", index=False, encoding="utf-8-sig")
    stage_summary.to_csv(args.output / "event_stage_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    spatial_summary.to_csv(args.output / "event_pair_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    change_summary.to_csv(args.output / "baseline_adjusted_bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    effects.to_csv(args.output / "low_vs_high_effect_sizes.csv", index=False, encoding="utf-8-sig")
    baseline_effects.to_csv(
        args.output / "baseline_adjusted_low_vs_high_effect_sizes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_stage_metrics(stage_table, stage_summary, args.output / "event_stage_bootstrap_intervals.png")
    plot_spatial_pairs(spatial_table, spatial_summary, args.output / "pressure_field_redistribution_statistics.png")
    plot_baseline_changes(
        change_table,
        change_summary,
        args.output / "baseline_adjusted_pressure_evolution.png",
    )
    plot_effect_sizes(effects, args.output / "low_vs_high_effect_sizes.png")
    write_summary(
        args.output / "bootstrap_event_analysis_summary_zh.md",
        stage_table,
        spatial_table,
        effects,
        baseline_effects,
        args.bootstrap_repeats,
    )
    metadata = {
        "included_specimens": int(stage_table["specimen_id"].nunique()),
        "rate_group_sizes": stage_table.groupby("rate_group", observed=True)["specimen_id"].nunique().to_dict(),
        "bootstrap_unit": "specimen",
        "bootstrap_repeats": args.bootstrap_repeats,
        "random_seed": args.seed,
        "event_stages_are_not_forced_chronological": True,
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
    }
    (args.output / "bootstrap_event_analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
