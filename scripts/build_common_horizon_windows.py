#!/usr/bin/env python3
"""Rebuild per-specimen causal-window labels on common cross-rate horizons."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def run_quiet(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {subprocess.list2cmdline(command)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    workspace = resolve(config_path.parent, config.get("workspace_root", "../..")).resolve()
    output_root = resolve(workspace, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    common_root = output_root / "common_horizon_windows"
    common_root.mkdir(parents=True, exist_ok=True)
    manifest_path = resolve(workspace, config["specimen_manifest"])
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        specimens = list(csv.DictReader(handle))

    horizons = [int(value) for value in config["target"]["prediction_horizons_ms"]]
    horizon_text = [str(value) for value in horizons]
    history_steps = int(config["sampling"]["history_steps"])
    primary_event = config["target"]["primary_event_id"]
    sensitivity_event = config["target"]["sensitivity_event_id"]
    scripts = workspace / "scripts"
    summary_rows: list[dict[str, Any]] = []

    for index, specimen in enumerate(specimens, start=1):
        specimen_id = specimen["specimen_id"]
        specimen_root = workspace / "data_interim" / specimen_id
        output_dir = common_root / specimen_id
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index:02d}/{len(specimens):02d}] {specimen_id}", flush=True)
        build_command = [
            sys.executable,
            str(scripts / "build_rkhs_event_windows.py"),
            "--unified-csv",
            str(specimen_root / "unified" / "unified_state_table.csv"),
            "--pressure-npz",
            str(specimen_root / "unified" / "pressure_fields_32x32.npz"),
            "--synchronization-json",
            str(specimen_root / "synchronization" / "synchronization_metadata.json"),
            "--machine-metadata-json",
            str(specimen_root / "machine" / "machine_metadata.json"),
            "--ground-truth-json",
            str(specimen_root / "ground_truth" / "ground_truth_events.json"),
            "--output-dir",
            str(output_dir),
            "--history-steps",
            str(history_steps),
            "--horizons-ms",
            *horizon_text,
        ]
        run_quiet(build_command)
        validation_path = output_dir / "validation_report.json"
        validate_command = [
            sys.executable,
            str(scripts / "validate_rkhs_event_windows.py"),
            "--manifest-csv",
            str(output_dir / "event_window_manifest.csv"),
            "--windows-npz",
            str(output_dir / "rkhs_causal_windows.npz"),
            "--metadata-json",
            str(output_dir / "event_window_metadata.json"),
            "--unified-csv",
            str(specimen_root / "unified" / "unified_state_table.csv"),
            "--output-json",
            str(validation_path),
        ]
        run_quiet(validate_command)
        metadata = load_json(output_dir / "event_window_metadata.json")
        validation = load_json(validation_path)
        events = metadata["target_definition"]["events"]
        row: dict[str, Any] = {
            "specimen_id": specimen_id,
            "rate_group": specimen["rate_group"],
            "unique_input_windows": int(metadata["counts"]["unique_input_windows"]),
            "manifest_rows": int(metadata["counts"]["event_window_manifest_rows"]),
            "validation_status": validation["status"],
        }
        for event_id in (primary_event, sensitivity_event, "machine_force_peak"):
            if event_id not in events:
                continue
            event = events[event_id]
            row[f"{event_id}_causal_windows"] = int(event["causal_window_count"])
            for horizon in horizons:
                counts = event["label_counts"][f"{horizon}ms"]
                prefix = f"{event_id}_{horizon}ms"
                row[f"{prefix}_positive"] = int(counts["positive"])
                row[f"{prefix}_negative"] = int(counts["negative"])
                row[f"{prefix}_ambiguous"] = int(counts["ambiguous"])
        summary_rows.append(row)

    fieldnames: list[str] = []
    for row in summary_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_root / "common_horizon_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    required_label_keys = {f"{horizon}ms" for horizon in horizons}
    all_pass = all(row["validation_status"] == "pass" for row in summary_rows)
    all_primary_labels_present = True
    for specimen in specimens:
        metadata = load_json(common_root / specimen["specimen_id"] / "event_window_metadata.json")
        labels = metadata["target_definition"]["events"][primary_event]["label_counts"]
        all_primary_labels_present &= required_label_keys == set(labels)
    report = {
        "schema_version": "1.0",
        "status": "pass" if all_pass and all_primary_labels_present else "fail",
        "specimen_count": len(summary_rows),
        "history_steps": history_steps,
        "common_prediction_horizons_ms": horizons,
        "primary_event_id": primary_event,
        "sensitivity_event_id": sensitivity_event,
        "all_window_validations_passed": all_pass,
        "all_primary_label_horizons_present": all_primary_labels_present,
        "single_specimen_outputs_preserved": True,
        "common_cache_root": str(common_root),
        "summary_csv": str(output_root / "common_horizon_summary.csv"),
    }
    (output_root / "common_horizon_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
