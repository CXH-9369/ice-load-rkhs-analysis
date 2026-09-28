#!/usr/bin/env python3
"""Validate RKHS causal-window outputs and write a compact audit report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def expected_label(endpoint: float, start: float, end: float, horizon_s: float) -> str:
    horizon_end = endpoint + horizon_s
    if horizon_end < start:
        return "0"
    if horizon_end >= end:
        return "1"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--unified-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_csv(args.manifest_csv)
    unified = read_csv(args.unified_csv)
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    arrays = np.load(args.windows_npz)

    errors: list[str] = []
    warnings: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    history_steps = int(metadata["window_definition"]["history_steps"])
    horizons_ms = [
        int(value) for value in metadata["window_definition"]["prediction_horizons_ms"]
    ]
    unique_windows = int(metadata["counts"]["unique_input_windows"])

    row_ids = [row["event_window_row_id"] for row in manifest]
    check(len(row_ids) == len(set(row_ids)), "Manifest event_window_row_id values are not unique")
    check(
        len(manifest) == int(metadata["counts"]["event_window_manifest_rows"]),
        "Manifest row count differs from metadata",
    )

    source_indices = arrays["source_state_indices_0_based"]
    endpoint_indices = arrays["endpoint_state_indices_0_based"]
    check(source_indices.shape == (unique_windows, history_steps), "Source-index shape mismatch")
    check(
        np.array_equal(np.diff(source_indices, axis=1), np.ones((unique_windows, history_steps - 1))),
        "At least one source-state window is not contiguous",
    )
    check(
        np.array_equal(source_indices[:, -1], endpoint_indices),
        "A source-state window includes data after its declared endpoint",
    )
    check(
        np.array_equal(arrays["endpoint_time_s"], np.asarray([float(unified[i]["unified_machine_time_s"]) for i in endpoint_indices])),
        "NPZ endpoint times do not match the unified table",
    )
    check(
        np.array_equal(arrays["endpoint_pressure_frame"], np.asarray([int(unified[i]["pressure_frame_number"]) for i in endpoint_indices])),
        "NPZ endpoint pressure frames do not match the unified table",
    )

    pressure_shape = tuple(int(value) for value in arrays["pressure_channels"].shape)
    mechanics_shape = tuple(int(value) for value in arrays["mechanics_features"].shape)
    scalar_shape = tuple(int(value) for value in arrays["pressure_scalar_features"].shape)
    check(
        pressure_shape == (unique_windows, history_steps, 3, 32, 32),
        f"Unexpected pressure tensor shape: {pressure_shape}",
    )
    check(mechanics_shape[:2] == (unique_windows, history_steps), "Mechanics tensor shape mismatch")
    check(scalar_shape[:2] == (unique_windows, history_steps), "Pressure-scalar tensor shape mismatch")
    check(np.isfinite(arrays["pressure_channels"]).all(), "Pressure tensor contains NaN or infinity")

    mechanics_valid = arrays["mechanics_valid_mask"]
    scalar_valid = arrays["pressure_scalar_valid_mask"]
    check(
        np.array_equal(mechanics_valid, np.isfinite(arrays["mechanics_features"])),
        "Mechanics validity mask does not match finite values",
    )
    check(
        np.array_equal(scalar_valid, np.isfinite(arrays["pressure_scalar_features"])),
        "Pressure-scalar validity mask does not match finite values",
    )

    event_counts: Counter[str] = Counter()
    latest_by_event: dict[str, dict[str, float | int]] = {}
    for row in manifest:
        event_id = row["event_id"]
        event_counts[event_id] += 1
        tensor_index = int(row["window_tensor_index_0_based"])
        endpoint = float(row["window_endpoint_time_s"])
        start = float(row["target_event_time_start_s"])
        end = float(row["target_event_time_end_s"])
        check(0 <= tensor_index < unique_windows, f"Invalid tensor index in {row['event_window_row_id']}")
        check(endpoint < start, f"Causality violation in {row['event_window_row_id']}")
        check(float(row["lead_to_event_interval_start_s"]) > 0, f"Non-positive lead in {row['event_window_row_id']}")
        check(row["input_uses_high_speed_features"] == "0", "High-speed feature entered an input row")
        check(row["causal_window_valid"] == "1", "Manifest contains an invalid causal window")
        check(
            np.isclose(endpoint, arrays["endpoint_time_s"][tensor_index], rtol=0, atol=1e-12),
            f"Manifest/NPZ endpoint mismatch in {row['event_window_row_id']}",
        )
        for horizon_ms in horizons_ms:
            actual = row[f"label_event_within_{horizon_ms}ms"]
            expected = expected_label(endpoint, start, end, horizon_ms / 1000.0)
            check(actual == expected, f"Incorrect {horizon_ms} ms label in {row['event_window_row_id']}")
        previous = latest_by_event.get(event_id)
        if previous is None or endpoint > float(previous["endpoint_time_s"]):
            latest_by_event[event_id] = {
                "endpoint_time_s": endpoint,
                "pressure_frame": int(row["window_endpoint_pressure_frame"]),
                "lead_to_event_start_s": float(row["lead_to_event_interval_start_s"]),
            }

    for event_id, summary in metadata["target_definition"]["events"].items():
        check(
            event_counts[event_id] == int(summary["causal_window_count"]),
            f"Event count differs from metadata for {event_id}",
        )

    missing_scalar_by_feature = {
        str(name): int(np.count_nonzero(~scalar_valid[:, :, index]))
        for index, name in enumerate(arrays["pressure_scalar_feature_names"])
        if np.count_nonzero(~scalar_valid[:, :, index])
    }
    if missing_scalar_by_feature:
        warnings.append(
            "Pressure-scalar NaNs are retained with validity masks; imputation must be fitted within each training fold."
        )

    report = {
        "schema_version": "1.0",
        "status": "pass" if not errors else "fail",
        "specimen_id": metadata["specimen_id"],
        "checks": {
            "strict_pre_event_causality": not any("Causality" in item for item in errors),
            "contiguous_history_indices": not any("not contiguous" in item for item in errors),
            "no_high_speed_input_features": not any("High-speed" in item for item in errors),
            "manifest_npz_endpoint_consistency": not any("endpoint mismatch" in item for item in errors),
            "label_recalculation_match": not any("label" in item.lower() for item in errors),
            "pressure_tensor_all_finite": bool(np.isfinite(arrays["pressure_channels"]).all()),
        },
        "counts": {
            "manifest_rows": len(manifest),
            "unique_input_windows": unique_windows,
            "events": dict(event_counts),
        },
        "tensor_shapes": {
            "pressure_channels": pressure_shape,
            "mechanics_features": mechanics_shape,
            "pressure_scalar_features": scalar_shape,
        },
        "latest_causal_endpoint_by_event": latest_by_event,
        "missing_pressure_scalar_values_by_feature": missing_scalar_by_feature,
        "warnings": warnings,
        "errors": errors,
    }
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
