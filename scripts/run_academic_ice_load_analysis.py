#!/usr/bin/env python3
"""Generate publication-oriented ice-load and RKHS evolution analyses.

The script treats specimens, rather than synchronized frames, as independent
experimental units.  It combines conventional ice-mechanics quantities with
RKHS state distances and kernel change points.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from run_rkhs_evolution_analysis import (
    COLORS,
    PRESSURE_FEATURES,
    RATE_ORDER,
    exact_mmd_test,
    robust_transform,
)


FACE_AREA_MM2 = 70.0 * 70.0
RATE_LABELS = {"low": "low", "medium": "medium", "high": "high"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def nearest_row(rows: list[dict[str, str]], target_time: float) -> dict[str, str]:
    return min(rows, key=lambda row: abs(as_float(row["time_s"]) - target_time))


def first_crossing_time(rows: list[dict[str, str]], target_fraction: float) -> float:
    for row in rows:
        if (
            row["pre_peak_loading_branch"] == "1"
            and row["pressure_shape_valid"] == "1"
            and as_float(row["force_fraction_of_specimen_peak"]) >= target_fraction
        ):
            return as_float(row["time_s"])
    return float("nan")


def unique_video_rows_before_cutoff(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["video_available"] != "1" or row["phase_family"] == "post_interpretable_cutoff":
            continue
        unique.setdefault(row["video_frame_number"], row)
    return sorted(unique.values(), key=lambda row: as_float(row["time_s"]))


def kernel_distance(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    squared = float(np.sum((x - y) ** 2))
    similarity = float(np.exp(-squared / (2.0 * bandwidth**2)))
    return float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))


def build_metrics(
    state_rows: list[dict[str, str]],
    summaries: list[dict[str, str]],
    transitions: list[dict[str, str]],
    morphology: list[dict[str, str]],
    binned: list[dict[str, str]],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    transition_by_id = {row["specimen_id"]: row for row in transitions}
    morphology_by_id = {row["specimen_id"]: row for row in morphology}
    center = np.asarray(metadata["robust_center"], dtype=float)
    scale = np.asarray(metadata["robust_iqr_scale"], dtype=float)
    bandwidth = float(metadata["rbf_bandwidth_kpca"])
    output: list[dict[str, object]] = []

    for summary in summaries:
        specimen = summary["specimen_id"]
        rows = [row for row in state_rows if row["specimen_id"] == specimen]
        bins = [row for row in binned if row["specimen_id"] == specimen]
        transition = transition_by_id[specimen]
        morphology_row = morphology_by_id[specimen]
        damage_time = as_float(summary["sustained_damage_time_s"])
        peak_time = as_float(summary["peak_force_time_s"])
        cp_fraction = as_float(transition["pressure_transition_force_fraction_of_peak"])
        cp_time = first_crossing_time(rows, cp_fraction)
        time_10 = first_crossing_time(rows, 0.10)
        loading_duration = peak_time - time_10
        lead_s = damage_time - cp_time
        lead_normalized = lead_s / loading_duration if loading_duration > 0 else float("nan")
        damage_row = nearest_row(rows, damage_time)
        peak_row = nearest_row(rows, peak_time)

        initial_vector = np.asarray([as_float(bins[0][name]) for name in PRESSURE_FEATURES])
        damage_vector = np.asarray([as_float(damage_row[name]) for name in PRESSURE_FEATURES])
        initial_scaled = robust_transform(initial_vector[None, :], center, scale)[0]
        damage_scaled = robust_transform(damage_vector[None, :], center, scale)[0]
        damage_state_distance = kernel_distance(initial_scaled, damage_scaled, bandwidth)

        video_rows = unique_video_rows_before_cutoff(rows)
        damage_areas = [as_float(row["video_damage_area_fraction"]) for row in video_rows]
        ejection_areas = [as_float(row["video_external_ejection_area_fraction"]) for row in video_rows]
        max_damage = float(np.nanmax(damage_areas)) if damage_areas else float("nan")
        max_ejection = float(np.nanmax(ejection_areas)) if ejection_areas else float("nan")

        peak_force = as_float(summary["peak_force_kn"])
        damage_force = as_float(summary["damage_force_kn"])
        output.append(
            {
                "specimen_id": specimen,
                "rate_group": summary["rate_group"],
                "nominal_rate_mm_min": as_float(summary["nominal_rate_mm_min"]),
                "actual_strain_rate_per_s": as_float(summary["actual_strain_rate_per_s"]),
                "peak_force_kn": peak_force,
                "peak_nominal_stress_mpa": peak_force * 1000.0 / FACE_AREA_MM2,
                "damage_force_kn": damage_force,
                "damage_nominal_stress_mpa": damage_force * 1000.0 / FACE_AREA_MM2,
                "damage_to_peak_force_ratio": as_float(summary["damage_to_peak_force_ratio"]),
                "damage_occurs_after_peak": int(damage_time > peak_time),
                "damage_time_minus_peak_time_s": damage_time - peak_time,
                "pressure_transition_force_fraction_of_peak": cp_fraction,
                "pressure_transition_time_s": cp_time,
                "pressure_transition_time_normalized_10pct_to_peak": (
                    (cp_time - time_10) / loading_duration if loading_duration > 0 else float("nan")
                ),
                "damage_time_normalized_10pct_to_peak": (
                    (damage_time - time_10) / loading_duration if loading_duration > 0 else float("nan")
                ),
                "pressure_transition_lead_to_damage_s": lead_s,
                "pressure_transition_lead_normalized_by_10pct_to_peak_duration": lead_normalized,
                "pressure_transition_precedes_damage": int(lead_s >= 0),
                "rkhs_distance_initial_to_damage_state": damage_state_distance,
                "damage_pressure_peak_concentration_factor": as_float(
                    damage_row["pressure_peak_to_machine_mean_pressure_ratio"]
                ),
                "damage_effective_pressure_area_fraction": as_float(
                    damage_row["pressure_effective_area_fraction_of_face"]
                ),
                "damage_pressure_entropy": as_float(damage_row["pressure_normalized_entropy"]),
                "damage_top10_load_fraction": as_float(
                    damage_row["pressure_top_10pct_cell_load_fraction"]
                ),
                "damage_film_to_machine_force_ratio": as_float(
                    damage_row["film_to_machine_force_ratio"]
                ),
                "damage_footprint_capture_fraction": as_float(
                    damage_row["pressure_footprint_capture_fraction"]
                ),
                "peak_footprint_capture_fraction": as_float(
                    peak_row["pressure_footprint_capture_fraction"]
                ),
                "damage_saturated_cell_count": int(
                    as_float(damage_row["pressure_saturated_cell_count_raw"])
                ),
                "damage_saturated_integral_fraction": as_float(
                    damage_row["pressure_saturated_fraction_corrected_integral"]
                ),
                "peak_saturated_cell_count": int(
                    as_float(peak_row["pressure_saturated_cell_count_raw"])
                ),
                "peak_saturated_integral_fraction": as_float(
                    peak_row["pressure_saturated_fraction_corrected_integral"]
                ),
                "maximum_visible_damage_area_fraction_pre_cutoff": max_damage,
                "maximum_external_ejection_area_fraction_pre_cutoff": max_ejection,
                "pressure_morphology_hsic": morphology_row[
                    "normalized_hsic_pressure_vs_visible_damage_ejection"
                ],
                "pressure_morphology_hsic_p": morphology_row["circular_shift_p_value"],
                "unique_interpretable_video_frames": int(
                    morphology_row["unique_interpretable_video_frames"]
                ),
                "unique_valid_fractal_frames": int(
                    morphology_row["unique_valid_fractal_frames"]
                ),
                "macro_failure_is_proxy": int(
                    summary["macro_failure_time_is_broad_response_proxy"]
                ),
            }
        )
    return output


def pairwise_metric_tests(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    tested = [
        "peak_nominal_stress_mpa",
        "damage_nominal_stress_mpa",
        "damage_to_peak_force_ratio",
        "pressure_transition_force_fraction_of_peak",
        "pressure_transition_lead_normalized_by_10pct_to_peak_duration",
        "rkhs_distance_initial_to_damage_state",
        "maximum_visible_damage_area_fraction_pre_cutoff",
    ]
    output: list[dict[str, object]] = []
    for feature in tested:
        for i, rate_a in enumerate(RATE_ORDER):
            for rate_b in RATE_ORDER[i + 1 :]:
                a = [float(row[feature]) for row in metrics if row["rate_group"] == rate_a]
                b = [float(row[feature]) for row in metrics if row["rate_group"] == rate_b]
                values = np.asarray(a + b, dtype=float)[:, None]
                score, p_value, permutations = exact_mmd_test(values, len(a))
                output.append(
                    {
                        "metric": feature,
                        "rate_group_a": rate_a,
                        "rate_group_b": rate_b,
                        "n_a": len(a),
                        "n_b": len(b),
                        "mmd2": score,
                        "exact_permutation_p_value": p_value,
                        "permutation_count": permutations,
                    }
                )
    return output


def rate_scatter(axis: plt.Axes, metrics: list[dict[str, object]], column: str, ylabel: str) -> None:
    for rate_index, rate in enumerate(RATE_ORDER):
        rows = [row for row in metrics if row["rate_group"] == rate]
        values = np.asarray([float(row[column]) for row in rows])
        x = rate_index + np.linspace(-0.10, 0.10, len(values))
        axis.scatter(x, values, color=COLORS[rate], s=48, zorder=3)
        axis.plot(
            [rate_index - 0.20, rate_index + 0.20],
            [np.median(values)] * 2,
            color="black",
            lw=1.7,
        )
    axis.set_xticks(range(3), RATE_ORDER)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.2)


def plot_strength(metrics: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for rate in RATE_ORDER:
        rows = [row for row in metrics if row["rate_group"] == rate]
        strain = np.asarray([float(row["actual_strain_rate_per_s"]) for row in rows])
        axes[0, 0].scatter(
            strain,
            [float(row["peak_nominal_stress_mpa"]) for row in rows],
            color=COLORS[rate],
            s=55,
            label=f"{rate} (n={len(rows)})",
        )
        axes[0, 1].scatter(
            strain,
            [float(row["damage_nominal_stress_mpa"]) for row in rows],
            color=COLORS[rate],
            s=55,
        )
    for axis, ylabel in [
        (axes[0, 0], "Peak nominal stress (MPa)"),
        (axes[0, 1], "Stress at sustained visible damage (MPa)"),
    ]:
        axis.set_xscale("log")
        axis.set_xlabel("Actual nominal strain rate (s$^{-1}$)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    rate_scatter(
        axes[1, 0], metrics, "damage_to_peak_force_ratio", "$F_{damage}/F_{peak}$"
    )
    for row in metrics:
        axes[1, 1].scatter(
            float(row["peak_nominal_stress_mpa"]),
            float(row["damage_nominal_stress_mpa"]),
            color=COLORS[str(row["rate_group"])],
            s=50,
        )
    maximum = max(float(row["peak_nominal_stress_mpa"]) for row in metrics) * 1.05
    axes[1, 1].plot([0, maximum], [0, maximum], color="black", ls="--", lw=1)
    axes[1, 1].set_xlim(0, maximum)
    axes[1, 1].set_ylim(0, maximum)
    axes[1, 1].set_xlabel("Peak nominal stress (MPa)")
    axes[1, 1].set_ylabel("Stress at sustained visible damage (MPa)")
    axes[1, 1].set_title("Damage timing relative to load capacity")
    axes[1, 1].grid(alpha=0.2)
    fig.suptitle("Ice load capacity and visible-damage onset")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_transition(metrics: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ordered = sorted(metrics, key=lambda row: (RATE_ORDER.index(str(row["rate_group"])), str(row["specimen_id"])))
    for y, row in enumerate(ordered):
        damage_position = float(row["damage_time_normalized_10pct_to_peak"])
        cp_position = float(row["pressure_transition_time_normalized_10pct_to_peak"])
        axes[0, 0].plot([cp_position, damage_position], [y, y], color=COLORS[str(row["rate_group"])], lw=2)
        axes[0, 0].scatter(cp_position, y, marker="o", color=COLORS[str(row["rate_group"])], s=35)
        axes[0, 0].scatter(damage_position, y, marker="^", facecolor="white", edgecolor="black", s=42, zorder=3)
    axes[0, 0].axvline(1.0, color="black", ls="--", lw=1, label="peak load time")
    axes[0, 0].set_yticks(range(len(ordered)))
    axes[0, 0].set_yticklabels([str(row["specimen_id"]).replace("-5-rigid-", "") for row in ordered], fontsize=8)
    axes[0, 0].set_xlabel("Normalized time: 10% load = 0, peak load = 1")
    axes[0, 0].set_title("Pressure transition (circle) to visible damage (triangle)")
    axes[0, 0].grid(axis="x", alpha=0.2)

    rate_scatter(
        axes[0, 1],
        metrics,
        "pressure_transition_force_fraction_of_peak",
        "Pressure-transition load / peak load",
    )
    rate_scatter(
        axes[1, 0],
        metrics,
        "pressure_transition_lead_normalized_by_10pct_to_peak_duration",
        "Transition-to-damage lead / loading duration",
    )
    axes[1, 0].axhline(0, color="black", ls="--", lw=1)
    rate_scatter(
        axes[1, 1],
        metrics,
        "rkhs_distance_initial_to_damage_state",
        "RKHS distance: initial to damage state",
    )
    fig.suptitle("RKHS pressure-state transition relative to visible damage")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_phase_portraits(binned: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pairs = [
        (
            "pressure_effective_area_fraction_of_face",
            "pressure_top_10pct_cell_load_fraction",
            "Effective pressure area / face area",
            "Load fraction in top 10% cells",
        ),
        (
            "pressure_normalized_entropy",
            "pressure_peak_to_machine_mean_pressure_ratio",
            "Normalized pressure entropy",
            "Peak / nominal mean pressure",
        ),
    ]
    for axis, (x_name, y_name, x_label, y_label) in zip(axes, pairs):
        for rate in RATE_ORDER:
            progress = sorted(
                {as_float(row["force_fraction_of_specimen_peak"]) for row in binned}
            )
            x_median, y_median = [], []
            for value in progress:
                selected = [
                    row
                    for row in binned
                    if row["rate_group"] == rate
                    and abs(as_float(row["force_fraction_of_specimen_peak"]) - value) < 1e-9
                ]
                x_median.append(float(np.median([as_float(row[x_name]) for row in selected])))
                y_median.append(float(np.median([as_float(row[y_name]) for row in selected])))
            axis.plot(x_median, y_median, color=COLORS[rate], lw=2.2, label=rate)
            scatter = axis.scatter(
                x_median,
                y_median,
                c=progress,
                cmap="viridis",
                vmin=0.1,
                vmax=1.0,
                s=25,
                edgecolor=COLORS[rate],
                linewidth=0.5,
            )
            axis.annotate(
                "",
                xy=(x_median[-1], y_median[-1]),
                xytext=(x_median[-3], y_median[-3]),
                arrowprops={"arrowstyle": "->", "color": COLORS[rate], "lw": 1.5},
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    colorbar = fig.colorbar(scatter, ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Machine force / specimen peak force")
    fig.suptitle("Pressure-localization phase portraits in the RKHS trajectory domain")
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.12, top=0.88, wspace=0.28)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_kernel_heatmap(
    metrics: list[dict[str, object]],
    binned: list[dict[str, str]],
    metadata: dict[str, object],
    output: Path,
) -> None:
    center = np.asarray(metadata["robust_center"], dtype=float)
    scale = np.asarray(metadata["robust_iqr_scale"], dtype=float)
    bandwidth = float(metadata["rbf_bandwidth_kpca"])
    ordered = sorted(metrics, key=lambda row: (RATE_ORDER.index(str(row["rate_group"])), str(row["specimen_id"])))
    grid = sorted({as_float(row["force_fraction_of_specimen_peak"]) for row in binned})
    matrix = []
    for metric in ordered:
        rows = [row for row in binned if row["specimen_id"] == metric["specimen_id"]]
        vectors = np.asarray([[as_float(row[name]) for name in PRESSURE_FEATURES] for row in rows])
        scaled = robust_transform(vectors, center, scale)
        matrix.append([kernel_distance(value, scaled[0], bandwidth) for value in scaled])
    matrix_array = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(15, 8))
    image = axis.imshow(matrix_array, aspect="auto", cmap="magma", vmin=0, vmax=np.nanmax(matrix_array))
    for y, metric in enumerate(ordered):
        cp = float(metric["pressure_transition_force_fraction_of_peak"])
        cp_x = int(np.argmin(np.abs(np.asarray(grid) - cp)))
        axis.scatter(cp_x, y, marker="o", facecolor="none", edgecolor="cyan", s=55, lw=1.2)
        damage_ratio = float(metric["damage_to_peak_force_ratio"])
        if int(metric["damage_occurs_after_peak"]):
            damage_x = len(grid) - 0.30
        else:
            damage_x = int(np.argmin(np.abs(np.asarray(grid) - damage_ratio)))
        axis.scatter(damage_x, y, marker="^", facecolor="white", edgecolor="black", s=45, lw=0.8)
    axis.set_xticks(range(len(grid)))
    axis.set_xticklabels([f"{value:.2f}" for value in grid], rotation=45)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([str(row["specimen_id"]).replace("-5-rigid-", "") for row in ordered])
    axis.set_xlabel("Machine force / specimen peak force")
    axis.set_ylabel("Specimen (low, medium, high from top to bottom)")
    axis.set_title("RKHS distance from the 10%-load pressure state\ncyan circle: kernel transition; white triangle: visible damage")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("RKHS state distance")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_morphology_quality(metrics: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    eligible = [row for row in metrics if row["pressure_morphology_hsic"] != ""]
    for row in eligible:
        axes[0, 0].scatter(
            float(row["pressure_transition_lead_normalized_by_10pct_to_peak_duration"]),
            float(row["pressure_morphology_hsic"]),
            s=30 + 1.5 * np.sqrt(float(row["unique_interpretable_video_frames"])),
            color=COLORS[str(row["rate_group"])],
            edgecolor="black" if float(row["pressure_morphology_hsic_p"]) <= 0.05 else "none",
        )
    axes[0, 0].axvline(0, color="black", ls="--", lw=1)
    axes[0, 0].set_xlabel("Normalized pressure-transition lead")
    axes[0, 0].set_ylabel("Pressure / visible-damage HSIC")
    axes[0, 0].set_title("Coupling strength; black rim: p <= 0.05")
    axes[0, 0].grid(alpha=0.2)
    rate_scatter(
        axes[0, 1],
        metrics,
        "maximum_visible_damage_area_fraction_pre_cutoff",
        "Maximum visible-damage area fraction",
    )
    rate_scatter(
        axes[1, 0],
        metrics,
        "maximum_external_ejection_area_fraction_pre_cutoff",
        "Maximum external-ejection area fraction",
    )
    ordered = sorted(metrics, key=lambda row: (RATE_ORDER.index(str(row["rate_group"])), str(row["specimen_id"])))
    axes[1, 1].bar(
        np.arange(len(ordered)),
        [int(row["unique_valid_fractal_frames"]) for row in ordered],
        color=[COLORS[str(row["rate_group"])] for row in ordered],
    )
    axes[1, 1].axhline(8, color="black", ls="--", lw=1)
    axes[1, 1].set_xticks(np.arange(len(ordered)))
    axes[1, 1].set_xticklabels(
        [str(row["specimen_id"]).replace("-5-rigid-", "") for row in ordered],
        rotation=55,
        ha="right",
    )
    axes[1, 1].set_ylabel("Unique valid fractal-dimension frames")
    axes[1, 1].set_title("Fractal metric coverage")
    fig.suptitle("Visible fracture morphology, pressure coupling, and quality control")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_pressure_measurement_quality(metrics: list[dict[str, object]], output: Path) -> None:
    ordered = sorted(
        metrics,
        key=lambda row: (RATE_ORDER.index(str(row["rate_group"])), str(row["specimen_id"])),
    )
    x = np.arange(len(ordered))
    labels = [str(row["specimen_id"]).replace("-5-rigid-", "") for row in ordered]
    colors = [COLORS[str(row["rate_group"])] for row in ordered]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes[0, 0].bar(x, [int(row["peak_saturated_cell_count"]) for row in ordered], color=colors)
    axes[0, 0].set_ylabel("Saturated cells at peak load")
    axes[0, 0].set_title("Sensor-range saturation count")
    axes[0, 1].bar(
        x,
        [float(row["peak_saturated_integral_fraction"]) for row in ordered],
        color=colors,
    )
    axes[0, 1].set_ylabel("Saturated contribution / pressure integral")
    axes[0, 1].set_title("Fraction of integrated pressure affected by saturation")
    axes[1, 0].bar(
        x,
        [float(row["damage_film_to_machine_force_ratio"]) for row in ordered],
        color=colors,
    )
    axes[1, 0].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 0].set_ylabel("Film force / machine force at visible damage")
    axes[1, 0].set_title("Cross-sensor force consistency (not expected to be identical)")
    axes[1, 1].bar(
        x,
        [float(row["damage_footprint_capture_fraction"]) for row in ordered],
        color=colors,
    )
    axes[1, 1].set_ylabel("Pressure captured inside reviewed 70 x 70 mm footprint")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_title("Reviewed footprint capture at visible damage")
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[1, :]:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right")
    fig.suptitle("Pressure-film measurement quality control")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def write_report(
    path: Path, metrics: list[dict[str, object]], tests: list[dict[str, object]]
) -> None:
    medians: dict[str, dict[str, float]] = {}
    for rate in RATE_ORDER:
        rows = [row for row in metrics if row["rate_group"] == rate]
        medians[rate] = {
            "peak_stress": float(np.median([float(row["peak_nominal_stress_mpa"]) for row in rows])),
            "damage_ratio": float(np.median([float(row["damage_to_peak_force_ratio"]) for row in rows])),
            "transition": float(np.median([float(row["pressure_transition_force_fraction_of_peak"]) for row in rows])),
            "lead": float(np.median([float(row["pressure_transition_lead_normalized_by_10pct_to_peak_duration"]) for row in rows])),
        }
    preceding = sum(int(row["pressure_transition_precedes_damage"]) for row in metrics)
    post_peak = sum(int(row["damage_occurs_after_peak"]) for row in metrics)
    saturated = sum(int(row["peak_saturated_cell_count"]) > 0 for row in metrics)
    force_consistency_flags = [
        str(row["specimen_id"])
        for row in metrics
        if float(row["damage_film_to_machine_force_ratio"]) > 1.2
        or float(row["damage_film_to_machine_force_ratio"]) < 0.1
    ]
    lines = [
        "# 专业冰载荷与RKHS演化分析 v1",
        "",
        "## 试件级力学结果",
        "",
        f"- 峰值名义压应力中位数：低速 {medians['low']['peak_stress']:.3f} MPa，中速 {medians['medium']['peak_stress']:.3f} MPa，高速 {medians['high']['peak_stress']:.3f} MPa。",
        f"- 首次持续可见破坏载荷与峰值载荷之比中位数：低速 {medians['low']['damage_ratio']:.3f}，中速 {medians['medium']['damage_ratio']:.3f}，高速 {medians['high']['damage_ratio']:.3f}。",
        f"- {post_peak}/16 组试件的首次持续可见破坏标注发生在压力机峰值之后；这类试件的事件载荷比不能直接解释为上升段起裂载荷。",
        "",
        "## RKHS压力状态变化",
        "",
        f"- 最强局部压力状态转换对应的 F/Fpeak 中位数：低速 {medians['low']['transition']:.2f}，中速 {medians['medium']['transition']:.2f}，高速 {medians['high']['transition']:.2f}。",
        f"- 压力状态转换在 {preceding}/16 组试件中早于首次持续可见破坏。",
        f"- 相对于10%载荷至峰值载荷持续时间的转换提前量中位数：低速 {medians['low']['lead']:.2f}，中速 {medians['medium']['lead']:.2f}，高速 {medians['high']['lead']:.2f}。",
        "",
        "## 压力数据质量",
        "",
        f"- {saturated}/16 组试件在峰值附近至少有一个达到传感器上限的单元；涉及峰值压力的结论必须同时查看饱和占比。",
        "- 首次持续可见破坏时薄膜合力/压力机载荷比超出0.1–1.2质量复核区间的试件："
        + ("、".join(force_consistency_flags) if force_consistency_flags else "无")
        + "。该标记用于识别快速卸载、同步差异或接触力未被薄膜完整捕获，不自动删除试件。",
        "- 薄膜积分力只表示接触区域压力响应，不替代压力机总载荷。",
        "- 分形维数继续仅用于满足有效性标志且具有足够独立视频帧的试件。",
        "",
        "## 试件级两组MMD检验",
        "",
        "结果见 `academic_pairwise_mmd.csv`。样本量尤其是低速组仅3个，因此p值用于描述证据强弱，不作为单一结论依据。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("data_interim/rkhs_evolution_v1/academic")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_rows = read_csv(args.root / "dataset" / "evolution_state_table.csv")
    summaries = read_csv(args.root / "dataset" / "specimen_event_summary.csv")
    transitions = read_csv(args.root / "analysis" / "pressure_transition_points.csv")
    morphology = read_csv(args.root / "analysis" / "pressure_morphology_hsic.csv")
    binned = read_csv(args.root / "analysis" / "balanced_loading_trajectories.csv")
    metadata = json.loads((args.root / "analysis" / "analysis_metadata.json").read_text(encoding="utf-8"))

    metrics = build_metrics(state_rows, summaries, transitions, morphology, binned, metadata)
    tests = pairwise_metric_tests(metrics)
    write_csv(args.output / "ice_load_specimen_metrics.csv", metrics)
    write_csv(args.output / "academic_pairwise_mmd.csv", tests)
    plot_strength(metrics, args.output / "ice_load_strength_and_damage.png")
    plot_transition(metrics, args.output / "rkhs_transition_vs_visible_damage.png")
    plot_phase_portraits(binned, args.output / "pressure_localization_phase_portraits.png")
    plot_kernel_heatmap(metrics, binned, metadata, args.output / "rkhs_state_distance_heatmap.png")
    plot_morphology_quality(metrics, args.output / "fracture_morphology_and_quality.png")
    plot_pressure_measurement_quality(
        metrics, args.output / "pressure_measurement_quality_control.png"
    )
    write_report(args.output / "academic_analysis_summary_zh.md", metrics, tests)
    print(f"Wrote academic ice-load analysis to {args.output}")


if __name__ == "__main__":
    main()
