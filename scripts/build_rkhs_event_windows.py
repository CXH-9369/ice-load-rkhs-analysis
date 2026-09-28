#!/usr/bin/env python3
"""Build causal event-centred input windows for the RKHS pipeline.

The output deliberately keeps raw, unnormalised measurements.  Fold-specific
normalisation and kernel length scales must be fitted later from training
specimens only.  High-speed-video features define targets but are never copied
into the causal input tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


MECHANICS_FEATURES = [
    "machine_force_kn",
    "machine_force_backward_rate_kn_per_s",
    "machine_deformation_from_first_record_mm",
    "machine_engineering_stress_actual_mpa",
    "machine_engineering_strain_actual_from_first_record_pct",
]

PRESSURE_SCALAR_FEATURES = [
    "pressure_vendor_force_n",
    "pressure_corrected_integral_n_equivalent",
    "pressure_corrected_peak_kpa",
    "pressure_area_ge_50kpa_mm2",
    "pressure_area_ge_250kpa_mm2",
    "pressure_area_ge_1000kpa_mm2",
    "pressure_centroid_x_specimen_mm",
    "pressure_centroid_y_toward_camera_mm",
    "pressure_rms_major_mm",
    "pressure_rms_minor_mm",
    "pressure_anisotropy",
    "pressure_effective_area_mm2",
    "pressure_normalized_entropy",
    "pressure_top_10pct_cell_load_fraction",
    "pressure_largest_cluster_area_ge_250kpa_mm2",
    "pressure_footprint_capture_fraction",
    "pressure_vendor_force_backward_rate_n_per_s",
    "pressure_corrected_integral_backward_rate_n_per_s",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | float | int | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def event_definitions(
    synchronization: dict[str, object], ground_truth: dict[str, object]
) -> list[dict[str, object]]:
    observations = {item["event_id"]: item for item in ground_truth["observations"]}

    machine_peak = float(synchronization["machine"]["peak_anchor_time_s"])
    sustained = observations["first_sustained_apparent_damage"]
    onset = observations.get("first_candidate_external_event", sustained)
    onset_interval = onset.get(
        "machine_time_interval_s",
        [
            float(observations["last_visually_intact"]["representative_machine_time_s"]),
            float(onset["representative_machine_time_s"]),
        ],
    )
    major = observations["major_fragmentation_pulse"]
    events = [
        {
            "event_id": "machine_force_peak",
            "event_class": "mechanical_peak",
            "time_start_s": machine_peak,
            "time_end_s": machine_peak,
            "source": "compression-machine peak plateau midpoint",
        },
        {
            "event_id": "first_visible_failure_interval",
            "event_class": onset.get("event_class", "apparent_local_damage_response"),
            "time_start_s": float(onset_interval[0]),
            "time_end_s": float(onset_interval[1]),
            "source": "confirmed high-speed first-detectable-damage interval",
        },
        {
            "event_id": "first_sustained_apparent_damage",
            "event_class": "apparent_surface_damage",
            "time_start_s": float(sustained["representative_machine_time_s"]),
            "time_end_s": float(sustained["representative_machine_time_s"]),
            "source": "high-speed human-review label",
        },
    ]
    if "proxy" not in str(major.get("event_class", "")):
        events.append({
            "event_id": "major_fragmentation_pulse",
            "event_class": "fragmentation",
            "time_start_s": float(major["representative_machine_time_s"]),
            "time_end_s": float(major["representative_machine_time_s"]),
            "source": "high-speed fragmentation label",
        })
    return events


def interval_horizon_label(
    endpoint_s: float, event_start_s: float, event_end_s: float, horizon_s: float
) -> int | str:
    """Return definite 0/1, or blank if the horizon cuts an uncertain interval."""
    horizon_end = endpoint_s + horizon_s
    if horizon_end < event_start_s:
        return 0
    if horizon_end >= event_end_s:
        return 1
    return ""


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_feature_matrix(
    rows: list[dict[str, str]], feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [[number(row[name]) for name in feature_names] for row in rows], dtype=np.float32
    )
    valid = np.isfinite(values)
    return values, valid


def plot_diagnostics(
    state_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, object]],
    events: list[dict[str, object]],
    horizons_ms: list[int],
    specimen_id: str,
    output: Path,
) -> None:
    time_s = np.asarray([number(row["unified_machine_time_s"]) for row in state_rows])
    force = np.asarray([number(row["machine_force_kn"]) for row in state_rows])
    unique_endpoint_times = sorted(
        {float(row["window_endpoint_time_s"]) for row in manifest_rows}
    )

    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.35, 1.0], hspace=0.34)
    axis_force = fig.add_subplot(grid[0, 0])
    axis_labels = fig.add_subplot(grid[1, 0])
    axis_counts = fig.add_subplot(grid[2, 0])

    axis_force.plot(time_s, force, color="#1f4e79", linewidth=1.8, label="Machine force")
    color_map = plt.get_cmap("tab10")
    colors = [color_map(index % 10) for index in range(len(events))]
    for event, color in zip(events, colors):
        start = float(event["time_start_s"])
        end = float(event["time_end_s"])
        if end > start:
            axis_force.axvspan(start, end, color=color, alpha=0.16)
        else:
            axis_force.axvline(start, color=color, linestyle="--", linewidth=1.1)
        eligible = [
            float(row["window_endpoint_time_s"])
            for row in manifest_rows
            if row["event_id"] == event["event_id"]
        ]
        latest_input = max(eligible)
        axis_force.scatter(
            [latest_input],
            [float(np.interp(latest_input, time_s, force))],
            s=48,
            color=color,
            marker="D",
            label=f"latest input: {event['event_id']}",
            zorder=3,
        )
    margin_s = max(0.03, 0.05 * (max(unique_endpoint_times) - min(unique_endpoint_times)))
    axis_force.set_xlim(
        max(float(np.min(time_s)), min(unique_endpoint_times) - margin_s),
        min(
            float(np.max(time_s)),
            max(max(float(event["time_end_s"]) for event in events), max(unique_endpoint_times))
            + margin_s,
        ),
    )
    axis_force.set_ylabel("Force (kN)")
    axis_force.set_title("Causal window endpoints before each target event")
    axis_force.legend(loc="upper left", ncol=2, fontsize=8)
    axis_force.grid(alpha=0.2)

    label_rows = []
    label_names = []
    endpoint_index = {value: i for i, value in enumerate(unique_endpoint_times)}
    for event in events:
        for horizon_ms in horizons_ms:
            values = np.full(len(unique_endpoint_times), np.nan, dtype=float)
            column = f"label_event_within_{horizon_ms}ms"
            for row in manifest_rows:
                if row["event_id"] != event["event_id"]:
                    continue
                value = row[column]
                if value != "":
                    values[endpoint_index[float(row["window_endpoint_time_s"])]] = float(value)
            label_rows.append(values)
            label_names.append(f"{event['event_id']} | {horizon_ms} ms")
    label_matrix = np.vstack(label_rows)
    masked = np.ma.masked_invalid(label_matrix)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#eeeeee")
    image = axis_labels.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    axis_labels.set_yticks(np.arange(len(label_names)), label_names, fontsize=8)
    axis_labels.set_xticks(
        np.arange(len(unique_endpoint_times)),
        [f"{value:.3f}" for value in unique_endpoint_times],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    axis_labels.set_xlabel("Window endpoint on machine time axis (s)")
    axis_labels.set_title("Event-within-horizon labels: red=0, green=1, grey=not eligible")
    fig.colorbar(image, ax=axis_labels, fraction=0.025, pad=0.02, ticks=[0, 1])

    category_names = []
    positive_counts = []
    negative_counts = []
    ambiguous_counts = []
    for event in events:
        for horizon_ms in horizons_ms:
            relevant = [row for row in manifest_rows if row["event_id"] == event["event_id"]]
            values = [row[f"label_event_within_{horizon_ms}ms"] for row in relevant]
            category_names.append(f"{event['event_id']}\n{horizon_ms} ms")
            positive_counts.append(sum(value == 1 for value in values))
            negative_counts.append(sum(value == 0 for value in values))
            ambiguous_counts.append(sum(value == "" for value in values))
    x = np.arange(len(category_names))
    axis_counts.bar(x, negative_counts, color="#d95f5f", label="Negative")
    axis_counts.bar(x, positive_counts, bottom=negative_counts, color="#70ad47", label="Positive")
    bottom = np.asarray(negative_counts) + np.asarray(positive_counts)
    axis_counts.bar(x, ambiguous_counts, bottom=bottom, color="#a6a6a6", label="Ambiguous")
    axis_counts.set_xticks(x, category_names, rotation=40, ha="right", fontsize=8)
    axis_counts.set_ylabel("Window count")
    axis_counts.set_title(f"Prototype label balance for specimen {specimen_id}")
    axis_counts.legend(loc="upper left", ncol=3, fontsize=8)
    axis_counts.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"{specimen_id} RKHS event-centred causal windows", fontsize=15, y=0.985
    )
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.12, top=0.90)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def plot_visible_onset_window(
    pressure_channels: np.ndarray,
    relative_times: np.ndarray,
    endpoint_times: np.ndarray,
    manifest_rows: list[dict[str, object]],
    specimen_id: str,
    output: Path,
) -> None:
    onset_rows = [
        row for row in manifest_rows if row["event_id"] == "first_visible_failure_interval"
    ]
    latest = max(onset_rows, key=lambda row: float(row["window_endpoint_time_s"]))
    window_index = int(latest["window_tensor_index_0_based"])
    pressure = pressure_channels[window_index, :, 0]
    rel_endpoint = relative_times[window_index]
    endpoint = float(endpoint_times[window_index])
    onset_start = float(latest["target_event_time_start_s"])
    vmax = float(np.max(pressure))

    fig = plt.figure(figsize=(15.5, 7.5))
    grid = fig.add_gridspec(
        2, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.34, hspace=0.36
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, column]) for column in range(4)],
            [fig.add_subplot(grid[1, column]) for column in range(4)],
        ]
    )
    colorbar_axis = fig.add_subplot(grid[:, 4])
    image = None
    for axis, matrix, relative in zip(axes.ravel(), pressure, rel_endpoint):
        absolute_time = endpoint + float(relative)
        image = axis.imshow(
            matrix,
            origin="upper",
            extent=[0, 70, 70, 0],
            cmap="turbo",
            vmin=0,
            vmax=vmax,
        )
        axis.set_title(
            f"t={absolute_time:.4f} s\nlead to onset={onset_start - absolute_time:.4f} s",
            fontsize=9,
        )
        axis.set_xlabel("x: camera left to right (mm)")
        axis.set_ylabel("y: toward camera (mm)")
    fig.colorbar(image, cax=colorbar_axis, label="Corrected pressure (kPa)")
    fig.suptitle(
        f"{specimen_id}: latest {pressure.shape[0]}-frame causal pressure window "
        "before first detectable damage\n"
        "All frames are inputs; the confirmed damage interval starts after the last panel",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.09, top=0.84)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-csv", type=Path, required=True)
    parser.add_argument("--pressure-npz", type=Path, required=True)
    parser.add_argument("--synchronization-json", type=Path, required=True)
    parser.add_argument("--machine-metadata-json", type=Path, required=True)
    parser.add_argument("--ground-truth-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--horizons-ms", type=int, nargs="+", default=[30, 60, 120])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.history_steps < 2:
        raise ValueError("history_steps must be at least 2")
    if any(value <= 0 for value in args.horizons_ms):
        raise ValueError("all prediction horizons must be positive")

    state_rows = read_csv(args.unified_csv)
    time_s = np.asarray([number(row["unified_machine_time_s"]) for row in state_rows])
    synchronization = json.loads(args.synchronization_json.read_text(encoding="utf-8"))
    machine_metadata = json.loads(args.machine_metadata_json.read_text(encoding="utf-8"))
    ground_truth = json.loads(args.ground_truth_json.read_text(encoding="utf-8"))
    events = event_definitions(synchronization, ground_truth)
    specimen_id = str(ground_truth["specimen_id"])
    crack_origin = ground_truth.get("crack_origin_label", {})
    spatial_origin_available = crack_origin.get("point_full_frame_px") is not None
    pressure_source = np.load(args.pressure_npz)

    for required in ("pressure_kpa", "grad_x_kpa_per_mm", "grad_y_toward_camera_kpa_per_mm"):
        if required not in pressure_source:
            raise KeyError(f"Missing pressure tensor field: {required}")
    if pressure_source["pressure_kpa"].shape[0] != len(state_rows):
        raise ValueError("Unified CSV rows and pressure tensor states do not match")

    mechanics, mechanics_valid = build_feature_matrix(state_rows, MECHANICS_FEATURES)
    pressure_scalars, pressure_scalars_valid = build_feature_matrix(
        state_rows, PRESSURE_SCALAR_FEATURES
    )

    eligible_by_event: dict[str, np.ndarray] = {}
    endpoint_union: set[int] = set()
    for event in events:
        event_start = float(event["time_start_s"])
        eligible = np.flatnonzero(
            (np.arange(len(state_rows)) >= args.history_steps - 1) & (time_s < event_start)
        )
        if eligible.size == 0:
            raise ValueError(f"No complete causal window is available for {event['event_id']}")
        eligible_by_event[str(event["event_id"])] = eligible
        endpoint_union.update(int(index) for index in eligible)

    endpoint_indices = np.asarray(sorted(endpoint_union), dtype=np.int32)
    endpoint_to_tensor = {int(value): index for index, value in enumerate(endpoint_indices)}
    state_window_indices = np.stack(
        [
            np.arange(index - args.history_steps + 1, index + 1, dtype=np.int32)
            for index in endpoint_indices
        ]
    )
    relative_time = time_s[state_window_indices] - time_s[endpoint_indices, None]

    pressure_channels_all = np.stack(
        [
            pressure_source["pressure_kpa"],
            pressure_source["grad_x_kpa_per_mm"],
            pressure_source["grad_y_toward_camera_kpa_per_mm"],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    pressure_channels = pressure_channels_all[state_window_indices]
    mechanics_windows = mechanics[state_window_indices]
    mechanics_valid_windows = mechanics_valid[state_window_indices]
    pressure_scalar_windows = pressure_scalars[state_window_indices]
    pressure_scalar_valid_windows = pressure_scalars_valid[state_window_indices]

    strain_rate = float(machine_metadata["loading_rate"]["actual_axial_strain_rate_per_s"])
    rate_log_epsilon = 1.0e-8
    manifest_rows: list[dict[str, object]] = []
    for event in events:
        event_id = str(event["event_id"])
        eligible = eligible_by_event[event_id]
        latest_endpoint = int(eligible[-1])
        for endpoint_index in eligible:
            endpoint_index = int(endpoint_index)
            endpoint = float(time_s[endpoint_index])
            event_start = float(event["time_start_s"])
            event_end = float(event["time_end_s"])
            row: dict[str, object] = {
                "specimen_id": specimen_id,
                "event_window_row_id": f"{specimen_id}-{event_id}-p{int(state_rows[endpoint_index]['pressure_frame_number']):04d}",
                "window_tensor_index_0_based": endpoint_to_tensor[endpoint_index],
                "event_id": event_id,
                "event_class": event["event_class"],
                "event_source": event["source"],
                "target_event_time_start_s": event_start,
                "target_event_time_end_s": event_end,
                "target_event_time_midpoint_s": 0.5 * (event_start + event_end),
                "window_endpoint_state_id": state_rows[endpoint_index]["state_id"],
                "window_endpoint_pressure_frame": int(
                    state_rows[endpoint_index]["pressure_frame_number"]
                ),
                "window_endpoint_time_s": endpoint,
                "window_history_start_time_s": float(
                    time_s[endpoint_index - args.history_steps + 1]
                ),
                "history_step_count": args.history_steps,
                "history_duration_s": float(
                    endpoint - time_s[endpoint_index - args.history_steps + 1]
                ),
                "lead_to_event_interval_start_s": event_start - endpoint,
                "lead_to_event_interval_end_s": event_end - endpoint,
                "latest_pressure_state_before_event": int(endpoint_index == latest_endpoint),
                "causal_window_valid": 1,
                "input_uses_high_speed_features": 0,
                "spatial_origin_label_available": int(spatial_origin_available),
                "representative_loading_rate_per_s": strain_rate,
                "log_loading_rate": float(np.log(strain_rate + rate_log_epsilon)),
            }
            for horizon_ms in args.horizons_ms:
                row[f"label_event_within_{horizon_ms}ms"] = interval_horizon_label(
                    endpoint, event_start, event_end, horizon_ms / 1000.0
                )
            manifest_rows.append(row)

    output_csv = args.output_dir / "event_window_manifest.csv"
    output_npz = args.output_dir / "rkhs_causal_windows.npz"
    output_json = args.output_dir / "event_window_metadata.json"
    output_png = args.output_dir / "event_window_diagnostics.png"
    output_example = args.output_dir / "visible_onset_causal_window.png"
    write_csv(output_csv, manifest_rows)
    np.savez_compressed(
        output_npz,
        pressure_channels=pressure_channels.astype(np.float32, copy=False),
        pressure_channel_names=np.asarray(
            ["pressure_kpa", "grad_x_kpa_per_mm", "grad_y_toward_camera_kpa_per_mm"]
        ),
        mechanics_features=mechanics_windows.astype(np.float32, copy=False),
        mechanics_valid_mask=mechanics_valid_windows,
        mechanics_feature_names=np.asarray(MECHANICS_FEATURES),
        pressure_scalar_features=pressure_scalar_windows.astype(np.float32, copy=False),
        pressure_scalar_valid_mask=pressure_scalar_valid_windows,
        pressure_scalar_feature_names=np.asarray(PRESSURE_SCALAR_FEATURES),
        relative_time_s=relative_time.astype(np.float32),
        source_state_indices_0_based=state_window_indices,
        endpoint_state_indices_0_based=endpoint_indices,
        endpoint_pressure_frame=np.asarray(
            [int(state_rows[index]["pressure_frame_number"]) for index in endpoint_indices],
            dtype=np.int32,
        ),
        endpoint_time_s=time_s[endpoint_indices].astype(np.float64),
        specimen_id=np.asarray([specimen_id] * len(endpoint_indices)),
        representative_loading_rate_per_s=np.full(
            len(endpoint_indices), strain_rate, dtype=np.float32
        ),
        log_loading_rate=np.full(
            len(endpoint_indices), np.log(strain_rate + rate_log_epsilon), dtype=np.float32
        ),
        footprint_overlap_fraction=pressure_source["footprint_overlap_fraction"].astype(
            np.float32, copy=False
        ),
        x_cell_center_specimen_mm=pressure_source["x_cell_center_specimen_mm"].astype(
            np.float32, copy=False
        ),
        y_cell_center_toward_camera_mm=pressure_source[
            "y_cell_center_toward_camera_mm"
        ].astype(np.float32, copy=False),
    )
    plot_diagnostics(
        state_rows, manifest_rows, events, args.horizons_ms, specimen_id, output_png
    )
    plot_visible_onset_window(
        pressure_channels,
        relative_time,
        time_s[endpoint_indices],
        manifest_rows,
        specimen_id,
        output_example,
    )

    event_summaries = {}
    for event in events:
        event_id = str(event["event_id"])
        event_rows = [row for row in manifest_rows if row["event_id"] == event_id]
        event_summaries[event_id] = {
            "target_time_interval_s": [
                float(event["time_start_s"]),
                float(event["time_end_s"]),
            ],
            "causal_window_count": len(event_rows),
            "latest_input_time_s": max(float(row["window_endpoint_time_s"]) for row in event_rows),
            "minimum_lead_to_interval_start_s": min(
                float(row["lead_to_event_interval_start_s"]) for row in event_rows
            ),
            "label_counts": {
                f"{horizon_ms}ms": {
                    "positive": sum(
                        row[f"label_event_within_{horizon_ms}ms"] == 1 for row in event_rows
                    ),
                    "negative": sum(
                        row[f"label_event_within_{horizon_ms}ms"] == 0 for row in event_rows
                    ),
                    "ambiguous": sum(
                        row[f"label_event_within_{horizon_ms}ms"] == "" for row in event_rows
                    ),
                }
                for horizon_ms in args.horizons_ms
            },
        }

    metadata = {
        "schema_version": "1.0",
        "specimen_id": specimen_id,
        "purpose": "prototype causal event-window inputs for later specimen-level LOSO RKHS modelling",
        "window_definition": {
            "primary_axis": "pressure-film frames",
            "history_steps": args.history_steps,
            "median_pressure_interval_s": float(np.median(np.diff(time_s))),
            "nominal_history_duration_s": float(
                np.median(time_s[args.history_steps - 1 :] - time_s[: -args.history_steps + 1])
            ),
            "prediction_horizons_ms": args.horizons_ms,
            "endpoint_rule": "strictly before the target event interval start",
            "causality_rule": "each tensor contains only the endpoint state and earlier states",
        },
        "input_definition": {
            "mechanics_features": MECHANICS_FEATURES,
            "pressure_scalar_features": PRESSURE_SCALAR_FEATURES,
            "pressure_channels": [
                "pressure_kpa",
                "grad_x_kpa_per_mm",
                "grad_y_toward_camera_kpa_per_mm",
            ],
            "pressure_grid": [32, 32],
            "dtype": "float32",
            "normalization": "not applied; estimate within each future LOSO training fold only",
            "video_features_used_as_input": False,
            "missing_value_policy": (
                "retain unavailable raw values as NaN and provide parallel boolean validity masks; "
                "fit any imputation value from the future LOSO training fold only"
            ),
        },
        "target_definition": {
            "events": event_summaries,
            "interval_label_policy": (
                "0 when the prediction horizon ends before the event interval; "
                "1 when it covers the full event interval; blank when it intersects only part of an uncertain interval"
            ),
            "spatial_origin_label": {
                "available": spatial_origin_available,
                "status": crack_origin.get("status", "not_provided"),
                "reason": crack_origin.get(
                    "reason",
                    "No spatial-origin annotation was supplied; absence is not a negative label.",
                ),
            },
        },
        "counts": {
            "unique_input_windows": len(endpoint_indices),
            "event_window_manifest_rows": len(manifest_rows),
            "missing_mechanics_values": int(np.count_nonzero(~mechanics_valid_windows)),
            "missing_pressure_scalar_values": int(
                np.count_nonzero(~pressure_scalar_valid_windows)
            ),
            "missing_pressure_channel_values": int(
                np.count_nonzero(~np.isfinite(pressure_channels))
            ),
        },
        "loading_rate": {
            "actual_axial_strain_rate_per_s": strain_rate,
            "log_epsilon": rate_log_epsilon,
            "log_loading_rate": float(np.log(strain_rate + rate_log_epsilon)),
        },
        "resource_profile": {
            "compute_profile": "laptop_safe",
            "tensor_dtype": "float32",
            "pressure_grid": [32, 32],
            "note": "small cached tensors; no raw-video access during RKHS training",
        },
        "outputs": {
            "manifest_csv": str(output_csv.resolve()),
            "causal_tensor_npz": str(output_npz.resolve()),
            "diagnostic_figure": str(output_png.resolve()),
            "visible_onset_example": str(output_example.resolve()),
        },
    }
    output_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
