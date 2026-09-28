#!/usr/bin/env python3
"""Exploratory, specimen-balanced RKHS analysis of ice-loading evolution."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


PRESSURE_FEATURES = [
    "pressure_peak_to_machine_mean_pressure_ratio",
    "pressure_area_ge_50kpa_fraction_of_face",
    "pressure_area_ge_250kpa_fraction_of_face",
    "pressure_centroid_x_fraction",
    "pressure_centroid_y_toward_camera_fraction",
    "pressure_anisotropy",
    "pressure_effective_area_fraction_of_face",
    "pressure_normalized_entropy",
    "pressure_top_10pct_cell_load_fraction",
    "pressure_largest_cluster_fraction_ge_250kpa",
    "pressure_footprint_capture_fraction",
    "film_to_machine_force_ratio",
]

RATE_ORDER = ["low", "medium", "high"]
COLORS = {"low": "#2b6cb0", "medium": "#d97706", "high": "#c53030"}

DISPLAY_NAMES = {
    "pressure_peak_to_machine_mean_pressure_ratio": "Peak / nominal mean pressure",
    "pressure_area_ge_50kpa_fraction_of_face": "Area >= 50 kPa / face area",
    "pressure_area_ge_250kpa_fraction_of_face": "Area >= 250 kPa / face area",
    "pressure_effective_area_fraction_of_face": "Effective pressure area / face area",
    "pressure_normalized_entropy": "Normalized pressure entropy",
    "pressure_top_10pct_cell_load_fraction": "Load fraction in top 10% cells",
}


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


def robust_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0)
    q25, q75 = np.nanpercentile(x, [25, 75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    return center, scale


def robust_transform(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (x - center) / scale
    result[~np.isfinite(result)] = 0.0
    return result


def squared_distances(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    if y is None:
        y = x
    result = (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T
    )
    return np.maximum(result, 0.0)


def median_bandwidth(x: np.ndarray) -> float:
    d2 = squared_distances(x)
    positive = d2[np.triu_indices(len(x), 1)]
    positive = positive[positive > 1e-14]
    return float(np.sqrt(np.median(positive))) if len(positive) else 1.0


def rbf_kernel(x: np.ndarray, bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    bandwidth = median_bandwidth(x) if bandwidth is None else bandwidth
    return np.exp(-squared_distances(x) / (2.0 * bandwidth**2)), bandwidth


def center_kernel(k: np.ndarray) -> np.ndarray:
    return k - k.mean(axis=0)[None, :] - k.mean(axis=1)[:, None] + k.mean()


def normalized_hsic(x: np.ndarray, y: np.ndarray) -> float:
    kx, _ = rbf_kernel(x)
    ky, _ = rbf_kernel(y)
    kxc, kyc = center_kernel(kx), center_kernel(ky)
    numerator = float(np.sum(kxc * kyc))
    denominator = float(np.sqrt(np.sum(kxc * kxc) * np.sum(kyc * kyc)))
    return numerator / denominator if denominator > 0 else float("nan")


def circular_shift_test(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    observed = normalized_hsic(x, y)
    null = np.asarray([normalized_hsic(x, np.roll(y, shift, axis=0)) for shift in range(1, len(y))])
    p_value = float((1 + np.count_nonzero(null >= observed)) / (1 + len(null)))
    return observed, p_value, len(null)


def biased_mmd2_from_kernel(kernel: np.ndarray, group_a: np.ndarray) -> float:
    group_b = ~group_a
    return float(
        kernel[np.ix_(group_a, group_a)].mean()
        + kernel[np.ix_(group_b, group_b)].mean()
        - 2.0 * kernel[np.ix_(group_a, group_b)].mean()
    )


def exact_mmd_test(x: np.ndarray, n_a: int) -> tuple[float, float, int]:
    kernel, _ = rbf_kernel(x)
    observed_mask = np.zeros(len(x), dtype=bool)
    observed_mask[:n_a] = True
    observed = biased_mmd2_from_kernel(kernel, observed_mask)
    values = []
    for combination in itertools.combinations(range(len(x)), n_a):
        mask = np.zeros(len(x), dtype=bool)
        mask[list(combination)] = True
        values.append(biased_mmd2_from_kernel(kernel, mask))
    null = np.asarray(values)
    p_value = float(np.count_nonzero(null >= observed - 1e-15) / len(null))
    return observed, p_value, len(null)


def binned_loading_trajectories(
    rows: list[dict[str, str]], grid: np.ndarray
) -> tuple[list[dict[str, object]], np.ndarray]:
    specimen_ids = list(dict.fromkeys(row["specimen_id"] for row in rows))
    records: list[dict[str, object]] = []
    matrices = []
    for specimen_id in specimen_ids:
        specimen_rows = [
            row
            for row in rows
            if row["specimen_id"] == specimen_id
            and row["pre_peak_loading_branch"] == "1"
            and row["machine_valid"] == "1"
            and row["pressure_shape_valid"] == "1"
        ]
        x = np.asarray([as_float(row["force_fraction_of_specimen_peak"]) for row in specimen_rows])
        features = np.asarray(
            [[as_float(row[name]) for name in PRESSURE_FEATURES] for row in specimen_rows]
        )
        trajectory = np.full((len(grid), len(PRESSURE_FEATURES)), np.nan)
        # Follow the causal loading path.  For each target load, interpolate at
        # its first chronological crossing; sorting all force values would mix
        # unloading/reloading states that happen to share the same force.
        for grid_index, target in enumerate(grid):
            crossings = np.flatnonzero(np.isfinite(x) & (x >= target))
            if len(crossings) == 0:
                continue
            right = int(crossings[0])
            if right == 0:
                trajectory[grid_index] = features[right]
                continue
            left = right - 1
            while left >= 0 and not np.isfinite(x[left]):
                left -= 1
            if left < 0 or x[right] <= x[left]:
                trajectory[grid_index] = features[right]
                continue
            fraction = float((target - x[left]) / (x[right] - x[left]))
            for feature_index in range(features.shape[1]):
                y0, y1 = features[left, feature_index], features[right, feature_index]
                if np.isfinite(y0) and np.isfinite(y1):
                    trajectory[grid_index, feature_index] = y0 + fraction * (y1 - y0)
                elif np.isfinite(y1):
                    trajectory[grid_index, feature_index] = y1
        coverage = np.mean(np.isfinite(trajectory), axis=1)
        if np.any(coverage < 0.8):
            raise ValueError(f"Insufficient loading-branch coverage for {specimen_id}")
        matrices.append(trajectory)
        rate = specimen_rows[0]["rate_group"]
        for index, progress in enumerate(grid):
            record: dict[str, object] = {
                "specimen_id": specimen_id,
                "rate_group": rate,
                "force_fraction_of_specimen_peak": float(progress),
            }
            record.update(
                {
                    feature: float(trajectory[index, j])
                    for j, feature in enumerate(PRESSURE_FEATURES)
                }
            )
            records.append(record)
    return records, np.asarray(matrices)


def kpca_coordinates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    kernel, bandwidth = rbf_kernel(x)
    centered = center_kernel(kernel)
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    coordinates = vectors[:, :2] * np.sqrt(values[:2])[None, :]
    explained = values[:2] / values.sum() if values.sum() > 0 else np.zeros(2)
    return coordinates, explained, bandwidth


def pressure_analysis(
    binned_rows: list[dict[str, object]], trajectories: np.ndarray, grid: np.ndarray
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    np.ndarray,
]:
    specimen_ids = list(dict.fromkeys(str(row["specimen_id"]) for row in binned_rows))
    rates = [
        str(next(row["rate_group"] for row in binned_rows if row["specimen_id"] == specimen))
        for specimen in specimen_ids
    ]
    flat = trajectories.reshape(-1, trajectories.shape[-1])
    center, scale = robust_fit(flat)
    standardized = robust_transform(flat, center, scale).reshape(trajectories.shape)

    hsic_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []
    for specimen_id, rate, trajectory in zip(specimen_ids, rates, standardized):
        score, p_value, permutations = circular_shift_test(grid[:, None], trajectory)
        hsic_rows.append(
            {
                "specimen_id": specimen_id,
                "rate_group": rate,
                "normalized_hsic_force_progress_vs_pressure_topology": score,
                "circular_shift_p_value": p_value,
                "circular_shift_count": permutations,
            }
        )
        candidates = []
        for boundary in range(3, len(grid) - 3):
            left = trajectory[boundary - 3 : boundary]
            right = trajectory[boundary : boundary + 3]
            joined = np.vstack([left, right])
            kernel, _ = rbf_kernel(joined)
            mask = np.zeros(6, dtype=bool)
            mask[:3] = True
            candidates.append((biased_mmd2_from_kernel(kernel, mask), boundary))
        score_cp, boundary = max(candidates)
        change_rows.append(
            {
                "specimen_id": specimen_id,
                "rate_group": rate,
                "maximum_local_mmd2": score_cp,
                "pressure_transition_force_fraction_of_peak": float(grid[boundary]),
                "left_window_start_force_fraction": float(grid[boundary - 3]),
                "right_window_end_force_fraction": float(grid[boundary + 2]),
            }
        )

    trajectory_vectors = standardized.reshape(len(specimen_ids), -1)
    shape_standardized = np.empty_like(standardized)
    for specimen_index, trajectory in enumerate(standardized):
        within_center, within_scale = robust_fit(trajectory)
        shape_standardized[specimen_index] = robust_transform(
            trajectory, within_center, within_scale
        )
    shape_vectors = shape_standardized.reshape(len(specimen_ids), -1)
    mmd_rows: list[dict[str, object]] = []
    for mode, vectors in [
        ("absolute_state_and_shape", trajectory_vectors),
        ("within_specimen_shape_only", shape_vectors),
    ]:
        for rate_a, rate_b in itertools.combinations(RATE_ORDER, 2):
            indices_a = [i for i, rate in enumerate(rates) if rate == rate_a]
            indices_b = [i for i, rate in enumerate(rates) if rate == rate_b]
            joined = vectors[indices_a + indices_b]
            score, p_value, permutations = exact_mmd_test(joined, len(indices_a))
            mmd_rows.append(
                {
                    "trajectory_mode": mode,
                    "rate_group_a": rate_a,
                    "rate_group_b": rate_b,
                    "specimen_count_a": len(indices_a),
                    "specimen_count_b": len(indices_b),
                    "trajectory_mmd2": score,
                    "exact_permutation_p_value": p_value,
                    "permutation_count": permutations,
                }
            )

    feature_mmd_rows: list[dict[str, object]] = []
    for rate_a, rate_b in itertools.combinations(RATE_ORDER, 2):
        indices_a = [i for i, rate in enumerate(rates) if rate == rate_a]
        indices_b = [i for i, rate in enumerate(rates) if rate == rate_b]
        pair_rows = []
        for feature_index, feature in enumerate(PRESSURE_FEATURES):
            feature_vectors = standardized[:, :, feature_index]
            joined = feature_vectors[indices_a + indices_b]
            score, p_value, permutations = exact_mmd_test(joined, len(indices_a))
            pair_rows.append(
                {
                    "rate_group_a": rate_a,
                    "rate_group_b": rate_b,
                    "pressure_feature": feature,
                    "trajectory_mmd2": score,
                    "exact_permutation_p_value": p_value,
                    "permutation_count": permutations,
                }
            )
        order = np.argsort([float(row["exact_permutation_p_value"]) for row in pair_rows])
        adjusted = np.ones(len(pair_rows), dtype=float)
        running = 1.0
        for reverse_rank in range(len(pair_rows) - 1, -1, -1):
            index = int(order[reverse_rank])
            rank = reverse_rank + 1
            raw = float(pair_rows[index]["exact_permutation_p_value"])
            running = min(running, raw * len(pair_rows) / rank)
            adjusted[index] = min(running, 1.0)
        for row, q_value in zip(pair_rows, adjusted):
            row["benjamini_hochberg_q_value_within_pair"] = float(q_value)
        feature_mmd_rows.extend(pair_rows)

    coordinates, explained, bandwidth = kpca_coordinates(standardized.reshape(-1, len(PRESSURE_FEATURES)))
    metadata = {
        "pressure_feature_names": PRESSURE_FEATURES,
        "robust_center": center.tolist(),
        "robust_iqr_scale": scale.tolist(),
        "rbf_bandwidth_kpca": bandwidth,
        "kpca_explained_kernel_eigenvalue_fraction_first_two": explained.tolist(),
        "trajectory_grid": grid.tolist(),
    }
    return hsic_rows, change_rows, mmd_rows, feature_mmd_rows, metadata, coordinates


def morphology_analysis(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    specimen_ids = list(dict.fromkeys(row["specimen_id"] for row in rows))
    for specimen_id in specimen_ids:
        candidates = [
            row
            for row in rows
            if row["specimen_id"] == specimen_id
            and row["video_available"] == "1"
            and row["phase_family"] != "post_interpretable_cutoff"
        ]
        unique: dict[str, dict[str, str]] = {}
        for row in candidates:
            unique.setdefault(row["video_frame_number"], row)
        ordered = sorted(unique.values(), key=lambda row: as_float(row["time_s"]))
        rate = ordered[0]["rate_group"] if ordered else next(
            row["rate_group"] for row in rows if row["specimen_id"] == specimen_id
        )
        if not ordered:
            output.append(
                {
                    "specimen_id": specimen_id,
                    "rate_group": rate,
                    "unique_interpretable_video_frames": 0,
                    "coupling_frame_count": 0,
                    "normalized_hsic_pressure_vs_visible_damage_ejection": "",
                    "circular_shift_p_value": "",
                    "circular_shift_count": 0,
                    "unique_valid_fractal_frames": 0,
                    "fractal_rkhs_eligible_ge_8_frames": 0,
                }
            )
            continue
        pressure = np.asarray(
            [[as_float(row[name]) for name in PRESSURE_FEATURES] for row in ordered]
        )
        morphology = np.asarray(
            [
                [
                    as_float(row["video_damage_area_fraction"]),
                    as_float(row["video_external_ejection_area_fraction"]),
                ]
                for row in ordered
            ]
        )
        valid = np.all(np.isfinite(pressure), axis=1) & np.all(np.isfinite(morphology), axis=1)
        valid_count = int(np.count_nonzero(valid))
        if valid_count >= 8:
            p_center, p_scale = robust_fit(pressure[valid])
            m_center, m_scale = robust_fit(morphology[valid])
            score, p_value, permutations = circular_shift_test(
                robust_transform(pressure[valid], p_center, p_scale),
                robust_transform(morphology[valid], m_center, m_scale),
            )
        else:
            score, p_value, permutations = float("nan"), float("nan"), 0
        fractal_frames = {
            row["video_frame_number"]
            for row in ordered
            if row["video_fractal_valid"] == "1"
            and np.isfinite(as_float(row["video_fractal_dimension_skeleton"]))
        }
        output.append(
            {
                "specimen_id": specimen_id,
                "rate_group": rate,
                "unique_interpretable_video_frames": len(ordered),
                "coupling_frame_count": valid_count,
                "normalized_hsic_pressure_vs_visible_damage_ejection": (
                    score if np.isfinite(score) else ""
                ),
                "circular_shift_p_value": p_value if np.isfinite(p_value) else "",
                "circular_shift_count": permutations,
                "unique_valid_fractal_frames": len(fractal_frames),
                "fractal_rkhs_eligible_ge_8_frames": int(len(fractal_frames) >= 8),
            }
        )
    return output


def plot_results(
    output: Path,
    binned_rows: list[dict[str, object]],
    coordinates: np.ndarray,
    hsic_rows: list[dict[str, object]],
    change_rows: list[dict[str, object]],
    morphology_rows: list[dict[str, object]],
    grid: np.ndarray,
    explained: list[float],
) -> None:
    specimen_ids = list(dict.fromkeys(str(row["specimen_id"]) for row in binned_rows))
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    coord = coordinates.reshape(len(specimen_ids), len(grid), 2)
    for i, specimen in enumerate(specimen_ids):
        rate = str(next(row["rate_group"] for row in binned_rows if row["specimen_id"] == specimen))
        axes[0, 0].plot(coord[i, :, 0], coord[i, :, 1], color=COLORS[rate], alpha=0.55, lw=1.2)
        axes[0, 0].scatter(coord[i, -1, 0], coord[i, -1, 1], color=COLORS[rate], s=16)
    axes[0, 0].set_xlabel(f"Kernel PC1 ({explained[0] * 100:.1f}% eigenvalue)")
    axes[0, 0].set_ylabel(f"Kernel PC2 ({explained[1] * 100:.1f}% eigenvalue)")
    axes[0, 0].set_title("Pressure-state trajectories; dot = peak load")

    for rate in RATE_ORDER:
        values = [
            float(row["normalized_hsic_force_progress_vs_pressure_topology"])
            for row in hsic_rows
            if row["rate_group"] == rate
        ]
        x = RATE_ORDER.index(rate) + np.linspace(-0.08, 0.08, len(values))
        axes[0, 1].scatter(x, values, color=COLORS[rate], s=45)
    axes[0, 1].set_xticks(range(3), RATE_ORDER)
    axes[0, 1].set_ylabel("Normalized HSIC")
    axes[0, 1].set_title("Load-progress / pressure-topology dependence")
    axes[0, 1].grid(axis="y", alpha=0.2)

    for rate in RATE_ORDER:
        values = [
            float(row["pressure_transition_force_fraction_of_peak"])
            for row in change_rows
            if row["rate_group"] == rate
        ]
        x = RATE_ORDER.index(rate) + np.linspace(-0.08, 0.08, len(values))
        axes[1, 0].scatter(x, values, color=COLORS[rate], s=45)
        axes[1, 0].plot(
            [RATE_ORDER.index(rate) - 0.18, RATE_ORDER.index(rate) + 0.18],
            [np.median(values)] * 2,
            color="black",
            lw=1.5,
        )
    axes[1, 0].set_xticks(range(3), RATE_ORDER)
    axes[1, 0].set_ylabel("Force / specimen peak force")
    axes[1, 0].set_title("Strongest local pressure-state transition")
    axes[1, 0].grid(axis="y", alpha=0.2)

    for rate in RATE_ORDER:
        eligible = [
            row
            for row in morphology_rows
            if row["rate_group"] == rate
            and row["normalized_hsic_pressure_vs_visible_damage_ejection"] != ""
        ]
        values = [float(row["normalized_hsic_pressure_vs_visible_damage_ejection"]) for row in eligible]
        x = RATE_ORDER.index(rate) + np.linspace(-0.08, 0.08, len(values)) if values else []
        axes[1, 1].scatter(x, values, color=COLORS[rate], s=45)
    axes[1, 1].set_xticks(range(3), RATE_ORDER)
    axes[1, 1].set_ylabel("Normalized HSIC")
    axes[1, 1].set_title("Pressure / visible-damage morphology coupling")
    axes[1, 1].grid(axis="y", alpha=0.2)
    fig.suptitle("Specimen-balanced RKHS evolution analysis")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def summarize_rate_features(
    binned_rows: list[dict[str, object]], grid: np.ndarray
) -> list[dict[str, object]]:
    specimen_ids = list(dict.fromkeys(str(row["specimen_id"]) for row in binned_rows))
    output: list[dict[str, object]] = []
    for rate in RATE_ORDER:
        rate_specimens = [
            specimen
            for specimen in specimen_ids
            if next(
                row["rate_group"]
                for row in binned_rows
                if row["specimen_id"] == specimen
            )
            == rate
        ]
        for feature in DISPLAY_NAMES:
            matrix = np.asarray(
                [
                    [
                        float(row[feature])
                        for row in binned_rows
                        if row["specimen_id"] == specimen
                    ]
                    for specimen in rate_specimens
                ]
            )
            specimen_means = matrix.mean(axis=1)
            for index, progress in enumerate(grid):
                output.append(
                    {
                        "rate_group": rate,
                        "specimen_count": len(rate_specimens),
                        "pressure_feature": feature,
                        "force_fraction_of_specimen_peak": float(progress),
                        "median": float(np.median(matrix[:, index])),
                        "q25": float(np.percentile(matrix[:, index], 25)),
                        "q75": float(np.percentile(matrix[:, index], 75)),
                        "median_of_specimen_trajectory_means": float(
                            np.median(specimen_means)
                        ),
                    }
                )
    return output


def plot_rate_feature_trajectories(
    binned_rows: list[dict[str, object]], grid: np.ndarray, output: Path
) -> None:
    specimen_ids = list(dict.fromkeys(str(row["specimen_id"]) for row in binned_rows))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for axis, feature in zip(axes.flat, DISPLAY_NAMES):
        for rate in RATE_ORDER:
            rate_specimens = [
                specimen
                for specimen in specimen_ids
                if next(
                    row["rate_group"]
                    for row in binned_rows
                    if row["specimen_id"] == specimen
                )
                == rate
            ]
            matrix = np.asarray(
                [
                    [
                        float(row[feature])
                        for row in binned_rows
                        if row["specimen_id"] == specimen
                    ]
                    for specimen in rate_specimens
                ]
            )
            median = np.median(matrix, axis=0)
            q25, q75 = np.percentile(matrix, [25, 75], axis=0)
            axis.fill_between(grid, q25, q75, color=COLORS[rate], alpha=0.12)
            axis.plot(
                grid,
                median,
                color=COLORS[rate],
                lw=2.2,
                label=f"{rate} (n={len(rate_specimens)})",
            )
        axis.set_title(DISPLAY_NAMES[feature])
        axis.set_xlabel("Machine force / specimen peak force")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Rate-dependent pressure evolution: specimen medians and interquartile ranges")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def median_by_rate(rows: list[dict[str, object]], column: str) -> dict[str, float]:
    return {
        rate: float(np.median([float(row[column]) for row in rows if row["rate_group"] == rate]))
        for rate in RATE_ORDER
    }


def write_summary(
    output: Path,
    hsic_rows: list[dict[str, object]],
    change_rows: list[dict[str, object]],
    mmd_rows: list[dict[str, object]],
    feature_mmd_rows: list[dict[str, object]],
    morphology_rows: list[dict[str, object]],
) -> None:
    load_hsic = median_by_rate(hsic_rows, "normalized_hsic_force_progress_vs_pressure_topology")
    transitions = median_by_rate(change_rows, "pressure_transition_force_fraction_of_peak")
    eligible_morphology = [
        row for row in morphology_rows if row["normalized_hsic_pressure_vs_visible_damage_ejection"] != ""
    ]
    minimum_resolution_count = sum(
        float(row["circular_shift_p_value"]) <= (1.0 / 19.0 + 1e-12)
        for row in hsic_rows
    )
    significant_morph = sum(
        float(row["circular_shift_p_value"]) <= 0.05 for row in eligible_morphology
    )
    fractal_eligible = sum(int(row["fractal_rkhs_eligible_ge_8_frames"]) for row in morphology_rows)
    lines = [
        "# RKHS evolution analysis v1",
        "",
        "## Scope",
        "",
        "The independent experimental unit is the specimen (n=16). Loading trajectories were interpolated to an equal 10%-to-100% peak-force grid, so low-rate videos do not receive extra weight merely because they contain more frames.",
        "",
        "## Initial results",
        "",
        f"- Median normalized HSIC is high, but a 19-state circular-shift test cannot attain p<0.05; {minimum_resolution_count}/16 specimens attained its minimum resolvable p=1/19=0.0526.",
        f"- Median normalized HSIC by rate: low={load_hsic['low']:.3f}, medium={load_hsic['medium']:.3f}, high={load_hsic['high']:.3f}.",
        f"- Median strongest pressure transition at F/Fpeak: low={transitions['low']:.2f}, medium={transitions['medium']:.2f}, high={transitions['high']:.2f}.",
        f"- Pressure/visible-damage morphology coupling evaluable in {len(eligible_morphology)}/16 specimens; significant in {significant_morph}/{len(eligible_morphology) if eligible_morphology else 0}.",
        f"- Fractal-dimension RKHS analysis has at least 8 unique valid frames in {fractal_eligible}/16 specimens; it remains a QC-qualified secondary analysis.",
        "",
        "## Rate-group trajectory MMD",
        "",
        "| Trajectory representation | Pair | MMD^2 | exact permutation p | n |",
        "|---|---|---:|---:|---:|",
    ]
    for row in mmd_rows:
        lines.append(
            f"| {row['trajectory_mode']} | {row['rate_group_a']} vs {row['rate_group_b']} | {float(row['trajectory_mmd2']):.4f} | {float(row['exact_permutation_p_value']):.4f} | {row['specimen_count_a']} vs {row['specimen_count_b']} |"
        )
    lines.extend(
        [
            "",
            "## Pressure features driving rate separation",
            "",
        ]
    )
    for rate_a, rate_b in itertools.combinations(RATE_ORDER, 2):
        significant_features = [
            str(row["pressure_feature"])
            for row in feature_mmd_rows
            if row["rate_group_a"] == rate_a
            and row["rate_group_b"] == rate_b
            and float(row["benjamini_hochberg_q_value_within_pair"]) <= 0.05
        ]
        lines.append(
            f"- {rate_a} vs {rate_b}: "
            + (", ".join(significant_features) if significant_features else "none after within-pair FDR control")
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- HSIC establishes nonlinear statistical dependence, not causality.",
            "- Circular shifts preserve the ordering and approximate temporal autocorrelation better than unrestricted frame permutation, but p-values are still exploratory with short trajectories.",
            "- Rate MMD uses one equal-weight trajectory vector per specimen; the low-rate group has only three specimens.",
            "- High-speed morphology is visible side-surface damage. It is not a direct measurement of internal three-dimensional cracking.",
            "- The pressure-film integral is a spatial/contact response and is not treated as the total machine load.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/dataset/evolution_state_table.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data_interim/rkhs_evolution_v1/analysis")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.dataset)
    # 210-10 first reaches the validated pressure-shape threshold at 9.39% of
    # peak load, making 10% the earliest fully observed common loading state.
    grid = np.linspace(0.10, 1.00, 19)
    binned_rows, trajectories = binned_loading_trajectories(rows, grid)
    (
        hsic_rows,
        change_rows,
        mmd_rows,
        feature_mmd_rows,
        metadata,
        coordinates,
    ) = pressure_analysis(binned_rows, trajectories, grid)
    morphology_rows = morphology_analysis(rows)
    rate_feature_rows = summarize_rate_features(binned_rows, grid)
    write_csv(args.output / "balanced_loading_trajectories.csv", binned_rows)
    write_csv(args.output / "load_pressure_hsic.csv", hsic_rows)
    write_csv(args.output / "pressure_transition_points.csv", change_rows)
    write_csv(args.output / "rate_group_trajectory_mmd.csv", mmd_rows)
    write_csv(args.output / "rate_group_feature_mmd.csv", feature_mmd_rows)
    write_csv(args.output / "pressure_morphology_hsic.csv", morphology_rows)
    write_csv(args.output / "rate_group_feature_trajectories.csv", rate_feature_rows)
    plot_results(
        args.output / "rkhs_evolution_overview.png",
        binned_rows,
        coordinates,
        hsic_rows,
        change_rows,
        morphology_rows,
        grid,
        metadata["kpca_explained_kernel_eigenvalue_fraction_first_two"],
    )
    plot_rate_feature_trajectories(
        binned_rows,
        grid,
        args.output / "rate_group_pressure_trajectories.png",
    )
    write_summary(
        args.output / "analysis_summary.md",
        hsic_rows,
        change_rows,
        mmd_rows,
        feature_mmd_rows,
        morphology_rows,
    )
    (args.output / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote RKHS evolution analysis to {args.output}")


if __name__ == "__main__":
    main()
