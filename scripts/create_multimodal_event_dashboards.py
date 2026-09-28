#!/usr/bin/env python3
"""Create synchronized load/pressure/high-speed/RKHS event dashboards."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_LOCAL_OPENCV = Path(__file__).resolve().parents[1] / ".deps" / "opencv"
if _LOCAL_OPENCV.exists():
    sys.path.insert(0, str(_LOCAL_OPENCV))

import cv2
import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from run_rkhs_evolution_analysis import (
    COLORS,
    PRESSURE_FEATURES,
    RATE_ORDER,
    robust_transform,
)


STAGE_COLORS = ["#6b7280", "#00a6d6", "#8b5cf6", "#dc2626"]


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


def first_crossing(rows: list[dict[str, str]], target: float) -> dict[str, str]:
    for row in rows:
        if (
            row["pre_peak_loading_branch"] == "1"
            and row["pressure_shape_valid"] == "1"
            and as_float(row["force_fraction_of_specimen_peak"]) >= target
        ):
            return row
    raise ValueError(f"No first crossing for {target}")


def nearest_valid(rows: list[dict[str, str]], target_time: float) -> dict[str, str]:
    valid = [row for row in rows if row["pressure_shape_valid"] == "1"]
    return min(valid, key=lambda row: abs(as_float(row["time_s"]) - target_time))


def video_mapping(specimen_root: Path) -> tuple[float, float]:
    metadata = json.loads(
        (specimen_root / "synchronization" / "synchronization_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    video = metadata["high_speed_video"]
    return float(video["physical_acquisition_rate_fps"]), float(
        video["offset_video_minus_machine_s"]
    )


def video_frame_number(machine_time: float, fps: float, offset: float) -> int:
    return int(round((machine_time + offset) * fps)) + 1


def read_video_crop(
    capture: cv2.VideoCapture, frame_number: int, bbox: list[int]
) -> np.ndarray | None:
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 1 or frame_number > frame_count:
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    ok, frame = capture.read()
    if not ok:
        return None
    x0, y0, x1, y1 = bbox
    crop = frame[y0:y1, x0:x1]
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def normalized_pressure(field: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    weighted = np.maximum(field, 0.0) * overlap
    return weighted / max(float(weighted.sum()), 1e-12) * 100.0


def kernel_distance_curve(
    binned_rows: list[dict[str, str]], metadata: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(metadata["robust_center"], dtype=float)
    scale = np.asarray(metadata["robust_iqr_scale"], dtype=float)
    bandwidth = float(metadata["rbf_bandwidth_kpca"])
    x = np.asarray([as_float(row["force_fraction_of_specimen_peak"]) for row in binned_rows])
    values = np.asarray(
        [[as_float(row[name]) for name in PRESSURE_FEATURES] for row in binned_rows]
    )
    scaled = robust_transform(values, center, scale)
    reference = scaled[0]
    squared = np.sum((scaled - reference) ** 2, axis=1)
    similarity = np.exp(-squared / (2.0 * bandwidth**2))
    return x, np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))


def stage_rows(
    rows: list[dict[str, str]], metric: dict[str, str]
) -> list[tuple[str, dict[str, str], float]]:
    transition_fraction = as_float(metric["pressure_transition_force_fraction_of_peak"])
    pre_fraction = max(0.10, transition_fraction - 0.15)
    damage_machine_time = as_float(rows[0]["time_s"]) - as_float(
        rows[0]["time_relative_sustained_damage_s"]
    )
    peak_row = max(rows, key=lambda row: as_float(row["force_kn"]))
    pre_row = first_crossing(rows, pre_fraction)
    transition_row = first_crossing(rows, transition_fraction)
    damage_row = nearest_valid(rows, damage_machine_time)
    peak_time = as_float(peak_row["time_s"])
    peak_pressure_row = nearest_valid(rows, peak_time)
    return [
        (f"Pre-transition\n{pre_fraction:.2f} F/Fpeak", pre_row, as_float(pre_row["time_s"])),
        (
            f"RKHS transition\n{transition_fraction:.2f} F/Fpeak",
            transition_row,
            as_float(transition_row["time_s"]),
        ),
        ("Sustained visible damage", damage_row, damage_machine_time),
        ("Machine peak load", peak_pressure_row, peak_time),
    ]


def select_representatives(
    binned: list[dict[str, str]], metadata: dict[str, object]
) -> dict[str, str]:
    center = np.asarray(metadata["robust_center"], dtype=float)
    scale = np.asarray(metadata["robust_iqr_scale"], dtype=float)
    specimen_ids = list(dict.fromkeys(row["specimen_id"] for row in binned))
    vectors: dict[str, np.ndarray] = {}
    rates: dict[str, str] = {}
    for specimen in specimen_ids:
        rows = [row for row in binned if row["specimen_id"] == specimen]
        values = np.asarray([[as_float(row[name]) for name in PRESSURE_FEATURES] for row in rows])
        vectors[specimen] = robust_transform(values, center, scale).ravel()
        rates[specimen] = rows[0]["rate_group"]
    selected: dict[str, str] = {}
    for rate in RATE_ORDER:
        candidates = [specimen for specimen in specimen_ids if rates[specimen] == rate]
        scores = {
            specimen: float(
                np.mean(
                    [
                        np.linalg.norm(vectors[specimen] - vectors[other])
                        for other in candidates
                        if other != specimen
                    ]
                )
            )
            for specimen in candidates
        }
        selected[rate] = min(scores, key=scores.get)
    return selected


def specimen_assets(
    workspace: Path,
    specimen: str,
    rows: list[dict[str, str]],
    metric: dict[str, str],
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, cv2.VideoCapture, list[int], float, float]:
    root = workspace / "data_interim" / specimen
    config = json.loads((workspace / "configs" / "specimens" / f"{specimen}.json").read_text(encoding="utf-8"))
    bbox = [int(value) for value in config["high_speed"]["roi_bbox_xyxy"]]
    fps, offset = video_mapping(root)
    capture = cv2.VideoCapture(str(root / "high_speed" / "high_speed_25fps.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open high-speed video for {specimen}")
    tensor = np.load(root / "unified" / "pressure_fields_32x32.npz")
    pressure = tensor["pressure_kpa"]
    overlap = tensor["footprint_overlap_fraction"]
    stages = []
    for label, row, anchor_time in stage_rows(rows, metric):
        pressure_time = as_float(row["time_s"])
        frame_number = video_frame_number(anchor_time, fps, offset)
        tensor_index = int(row["pressure_field_tensor_index_0_based"])
        stages.append(
            {
                "label": label,
                "row": row,
                "machine_time_s": anchor_time,
                "pressure_time_s": pressure_time,
                "video_frame_number": frame_number,
                "video_image": read_video_crop(capture, frame_number, bbox),
                "pressure_raw": pressure[tensor_index].copy(),
                "pressure_percent": normalized_pressure(pressure[tensor_index], overlap),
            }
        )
    tensor.close()
    return stages, pressure, overlap, capture, bbox, fps, offset


def plot_specimen_dashboard(
    specimen: str,
    rate: str,
    rows: list[dict[str, str]],
    binned_rows: list[dict[str, str]],
    metric: dict[str, str],
    metadata: dict[str, object],
    stages: list[dict[str, object]],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(18, 12))
    grid = GridSpec(3, 4, figure=figure, height_ratios=[1.0, 1.15, 0.85], hspace=0.30, wspace=0.18)
    vmax = float(np.percentile(np.asarray([stage["pressure_percent"] for stage in stages]), 99.5))
    pressure_image = None
    for column, (stage, color) in enumerate(zip(stages, STAGE_COLORS)):
        axis = figure.add_subplot(grid[0, column])
        pressure_image = axis.imshow(
            stage["pressure_percent"],
            origin="lower",
            extent=[-35, 35, -35, 35],
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        saturated = np.asarray(stage["pressure_raw"]) >= 12880.0 - 1e-3
        if np.any(saturated):
            axis.contour(
                np.linspace(-34, 34, 32),
                np.linspace(-34, 34, 32),
                saturated.astype(float),
                levels=[0.5],
                colors="cyan",
                linewidths=0.7,
            )
        axis.set_title(str(stage["label"]), color=color, fontweight="bold")
        axis.set_xlabel("x (mm)")
        if column == 0:
            axis.set_ylabel("y toward camera (mm)")
        axis.set_xticks([-35, 0, 35])
        axis.set_yticks([-35, 0, 35])

        video_axis = figure.add_subplot(grid[1, column])
        if stage["video_image"] is None:
            video_axis.set_facecolor("#eeeeee")
            video_axis.text(
                0.5,
                0.5,
                "High-speed frame\nnot recorded at this time",
                ha="center",
                va="center",
                transform=video_axis.transAxes,
                fontsize=11,
            )
        else:
            video_axis.imshow(stage["video_image"])
        video_axis.set_title(
            f"frame {stage['video_frame_number']} | machine t={stage['machine_time_s']:.4f} s",
            fontsize=10,
        )
        video_axis.axis("off")
    if pressure_image is not None:
        colorbar_axis = figure.add_axes([0.92, 0.675, 0.012, 0.20])
        colorbar = figure.colorbar(pressure_image, cax=colorbar_axis)
        colorbar.set_label("Cell share of contact load (%)")

    time = np.asarray([as_float(row["time_s"]) for row in rows])
    force_fraction = np.asarray([as_float(row["force_fraction_of_specimen_peak"]) for row in rows])
    peak_index = int(np.nanargmax(force_fraction))
    time10 = as_float(first_crossing(rows, 0.10)["time_s"])
    peak_time = time[peak_index]
    duration = peak_time - time10
    normalized_time = (time - time10) / duration
    load_axis = figure.add_subplot(grid[2, :2])
    load_axis.plot(normalized_time, force_fraction, color="#1f2937", lw=1.5)
    for stage, color in zip(stages, STAGE_COLORS):
        x = (float(stage["machine_time_s"]) - time10) / duration
        load_axis.axvline(x, color=color, lw=1.4, alpha=0.9)
    load_axis.axhline(1.0, color="black", ls="--", lw=0.8)
    load_axis.set_xlabel("Normalized time: 10% load = 0, peak load = 1")
    load_axis.set_ylabel("Machine force / specimen peak force")
    load_axis.set_title("Mechanical load history and multimodal sampling times")
    load_axis.grid(alpha=0.2)

    distance_axis = figure.add_subplot(grid[2, 2:])
    progress, distance = kernel_distance_curve(binned_rows, metadata)
    distance_axis.plot(progress, distance, color="#111827", lw=2)
    transition_fraction = as_float(metric["pressure_transition_force_fraction_of_peak"])
    distance_axis.axvline(transition_fraction, color=STAGE_COLORS[1], lw=1.6)
    damage_fraction = as_float(metric["damage_to_peak_force_ratio"])
    marker_x = 1.0 if metric["damage_occurs_after_peak"] == "1" else damage_fraction
    marker_y = float(np.interp(marker_x, progress, distance))
    distance_axis.scatter(marker_x, marker_y, marker="^", color=STAGE_COLORS[2], s=65, zorder=3)
    distance_axis.set_xlabel("Machine force / specimen peak force")
    distance_axis.set_ylabel("RKHS distance from 10%-load pressure state")
    distance_axis.set_title("Nonlinear pressure-state evolution")
    distance_axis.grid(alpha=0.2)
    figure.suptitle(
        f"{specimen} | {rate} rate | synchronized ice-load evolution dashboard\n"
        "event-defined columns (not necessarily chronological); pressure-map lower edge faces the camera; "
        "cyan contours denote saturated cells",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.055, right=0.90, bottom=0.06, top=0.91)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_representative_comparison(
    representatives: dict[str, str],
    assets: dict[str, list[dict[str, object]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(17, 19))
    all_pressure = np.asarray(
        [stage["pressure_percent"] for specimen in representatives.values() for stage in assets[specimen]]
    )
    vmax = float(np.percentile(all_pressure, 99.5))
    image = None
    for rate_index, rate in enumerate(RATE_ORDER):
        specimen = representatives[rate]
        for column, stage in enumerate(assets[specimen]):
            pressure_axis = axes[2 * rate_index, column]
            image = pressure_axis.imshow(
                stage["pressure_percent"],
                origin="lower",
                extent=[-35, 35, -35, 35],
                cmap="inferno",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
            if rate_index == 0:
                pressure_axis.set_title(str(stage["label"]), fontsize=11)
            if column == 0:
                pressure_axis.set_ylabel(f"{rate}: {specimen}\npressure y (mm)")
            pressure_axis.set_xticks([-35, 0, 35])
            pressure_axis.set_yticks([-35, 0, 35])
            video_axis = axes[2 * rate_index + 1, column]
            if stage["video_image"] is None:
                video_axis.set_facecolor("#eeeeee")
                video_axis.text(0.5, 0.5, "not recorded", ha="center", va="center", transform=video_axis.transAxes)
            else:
                video_axis.imshow(stage["video_image"])
            if column == 0:
                video_axis.set_ylabel("side view")
            video_axis.axis("off")
    assert image is not None
    colorbar_axis = fig.add_axes([0.925, 0.39, 0.012, 0.24])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Cell share of contact load (%)")
    fig.suptitle(
        "Objectively selected rate-group medoids: contact pressure and visible side-surface evolution",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.974,
        "Event-defined columns are not necessarily chronological; specimen dashboards report the true machine times",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.035, top=0.935, wspace=0.12, hspace=0.18)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/multimodal_dashboards"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state_rows = read_csv(args.root / "dataset" / "evolution_state_table.csv")
    binned = read_csv(args.root / "analysis" / "balanced_loading_trajectories.csv")
    metrics = read_csv(args.root / "academic" / "ice_load_specimen_metrics.csv")
    metadata = json.loads((args.root / "analysis" / "analysis_metadata.json").read_text(encoding="utf-8"))
    metric_by_id = {row["specimen_id"]: row for row in metrics}
    representatives = select_representatives(binned, metadata)
    index_rows: list[dict[str, object]] = []
    cached_assets: dict[str, list[dict[str, object]]] = {}

    specimen_ids = list(dict.fromkeys(row["specimen_id"] for row in state_rows))
    for specimen in specimen_ids:
        rows = [row for row in state_rows if row["specimen_id"] == specimen]
        binned_rows = [row for row in binned if row["specimen_id"] == specimen]
        metric = metric_by_id[specimen]
        stages, _, _, capture, _, _, _ = specimen_assets(
            args.workspace, specimen, rows, metric
        )
        cached_assets[specimen] = stages
        plot_specimen_dashboard(
            specimen,
            str(metric["rate_group"]),
            rows,
            binned_rows,
            metric,
            metadata,
            stages,
            args.output / f"{specimen}_multimodal_dashboard.png",
        )
        for stage in stages:
            index_rows.append(
                {
                    "specimen_id": specimen,
                    "rate_group": metric["rate_group"],
                    "representative_rate_medoid": int(
                        representatives[str(metric["rate_group"])] == specimen
                    ),
                    "stage": str(stage["label"]).replace("\n", " "),
                    "machine_time_s": stage["machine_time_s"],
                    "pressure_frame_time_s": stage["pressure_time_s"],
                    "pressure_frame_number": stage["row"]["pressure_frame_number"],
                    "video_frame_number": stage["video_frame_number"],
                    "video_frame_available": int(stage["video_image"] is not None),
                    "force_fraction_of_specimen_peak": stage["row"][
                        "force_fraction_of_specimen_peak"
                    ],
                }
            )
        capture.release()
    write_csv(args.output / "multimodal_stage_index.csv", index_rows)
    (args.output / "representative_specimens.json").write_text(
        json.dumps(representatives, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_representative_comparison(
        representatives,
        cached_assets,
        args.output.parent / "representative_multimodal_comparison.png",
    )
    print(json.dumps(representatives, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
