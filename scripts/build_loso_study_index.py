#!/usr/bin/env python3
"""Build and validate the specimen-level index for 16-fold LOSO modelling."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_event(metadata: dict[str, Any], event_id: str) -> dict[str, Any]:
    events = metadata.get("target_definition", {}).get("events", {})
    if event_id not in events:
        raise KeyError(f"Missing target event {event_id}")
    return events[event_id]


def read_scope(workspace: Path, config: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    manifest_path = resolve(workspace, config["specimen_manifest"])
    exclusion_path = resolve(workspace, config["exclusion_manifest"])
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        specimens = list(csv.DictReader(handle))
    with exclusion_path.open("r", encoding="utf-8-sig", newline="") as handle:
        excluded = [row["specimen_id"] for row in csv.DictReader(handle)]
    return specimens, excluded


def build_specimen_index(
    workspace: Path,
    manifest_rows: list[dict[str, str]],
    primary_event_id: str,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    rows: list[dict[str, Any]] = []
    configured_horizons: dict[str, list[int]] = {}
    for manifest_row in manifest_rows:
        specimen_id = manifest_row["specimen_id"]
        root = workspace / "data_interim" / specimen_id
        specimen_config = load_json(workspace / "configs" / "specimens" / f"{specimen_id}.json")
        machine = load_json(root / "machine" / "machine_metadata.json")
        windows = load_json(root / "rkhs_windows" / "event_window_metadata.json")
        validation = load_json(root / "rkhs_windows" / "validation_report.json")
        ground_truth = load_json(root / "ground_truth" / "ground_truth_events.json")
        event = first_event(windows, primary_event_id)
        horizons = [int(value) for value in specimen_config["model"]["prediction_horizons_ms"]]
        configured_horizons[specimen_id] = horizons
        label_counts = event.get("label_counts", {})
        rows.append(
            {
                "specimen_id": specimen_id,
                "rate_group": manifest_row["rate_group"],
                "nominal_rate_mm_min": float(manifest_row["nominal_rate_mm_min"]),
                "actual_strain_rate_per_s": float(machine["loading_rate"]["actual_axial_strain_rate_per_s"]),
                "log_actual_strain_rate": math.log(
                    float(machine["loading_rate"]["actual_axial_strain_rate_per_s"]) + 1.0e-8
                ),
                "machine_sampling_rate_hz": float(machine["time"]["sampling_rate_hz"]),
                "camera_acquisition_fps": float(specimen_config["high_speed"]["acquisition_fps"]),
                "camera_frame_interval_ms": 1000.0 / float(specimen_config["high_speed"]["acquisition_fps"]),
                "event_time_start_s": float(event["target_time_interval_s"][0]),
                "event_time_end_s": float(event["target_time_interval_s"][1]),
                "event_interval_width_s": float(event["target_time_interval_s"][1])
                - float(event["target_time_interval_s"][0]),
                "available_unique_windows": int(windows["counts"]["unique_input_windows"]),
                "primary_event_causal_windows": int(event["causal_window_count"]),
                "current_prediction_horizons_ms": ";".join(str(value) for value in horizons),
                "current_label_counts_json": json.dumps(label_counts, ensure_ascii=False, separators=(",", ":")),
                "window_validation_status": validation["status"],
                "ground_truth_status": ground_truth["annotation_status"],
                "window_npz": str(root / "rkhs_windows" / "rkhs_causal_windows.npz"),
                "window_manifest_csv": str(root / "rkhs_windows" / "event_window_manifest.csv"),
                "event_metadata_json": str(root / "rkhs_windows" / "event_window_metadata.json"),
            }
        )
    return rows, configured_horizons


def build_folds(
    specimen_rows: list[dict[str, Any]],
    maximum_training_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ids = [row["specimen_id"] for row in specimen_rows]
    rate_by_id = {row["specimen_id"]: row["rate_group"] for row in specimen_rows}
    windows_by_id = {row["specimen_id"]: int(row["available_unique_windows"]) for row in specimen_rows}
    csv_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    for fold_number, test_id in enumerate(ids, start=1):
        train_ids = [specimen_id for specimen_id in ids if specimen_id != test_id]
        group_counts = Counter(rate_by_id[specimen_id] for specimen_id in train_ids)
        raw_train_windows = sum(windows_by_id[specimen_id] for specimen_id in train_ids)
        specimen_quota = maximum_training_samples // len(train_ids)
        fold_id = f"fold_{fold_number:02d}"
        csv_rows.append(
            {
                "fold_id": fold_id,
                "test_specimen_id": test_id,
                "test_rate_group": rate_by_id[test_id],
                "train_specimen_count": len(train_ids),
                "train_low_count": group_counts.get("low", 0),
                "train_medium_count": group_counts.get("medium", 0),
                "train_high_count": group_counts.get("high", 0),
                "raw_training_unique_windows": raw_train_windows,
                "maximum_training_samples": maximum_training_samples,
                "nominal_specimen_balanced_quota": specimen_quota,
                "test_available_unique_windows": windows_by_id[test_id],
                "train_specimen_ids": ";".join(train_ids),
            }
        )
        json_rows.append(
            {
                "fold_id": fold_id,
                "test_specimen_id": test_id,
                "test_rate_group": rate_by_id[test_id],
                "train_specimen_ids": train_ids,
                "train_rate_group_counts": dict(group_counts),
                "raw_training_unique_windows": raw_train_windows,
                "maximum_training_samples": maximum_training_samples,
                "sampling_policy": {
                    "specimen_balanced_nominal_quota": specimen_quota,
                    "rate_balanced_group_budget": maximum_training_samples // len(group_counts),
                    "shortfall_policy": "redistribute unused quota deterministically within the training fold only",
                },
            }
        )
    return csv_rows, json_rows


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# 16 组 LOSO 数据装配就绪报告",
        "",
        f"- 状态：`{report['status']}`",
        f"- 外层折数：`{report['counts']['folds']}`",
        f"- 正式试件：`{report['counts']['specimens']}`",
        f"- 速率组：`{report['counts']['rate_groups']}`",
        f"- 公共预测时域：`{report['common_prediction_horizons_ms']} ms`",
        f"- 每折最大训练样本：`{report['resource_plan']['maximum_training_samples_per_fold']}`",
        f"- 最大原始训练窗口数：`{report['resource_plan']['maximum_raw_training_windows']}`",
        f"- 预计单个 float32 Gram 矩阵：`{report['resource_plan']['single_kernel_mib_at_cap']:.1f} MiB`",
        "",
        "## 防泄漏检查",
        "",
    ]
    for name, value in report["leakage_checks"].items():
        lines.append(f"- {name}：`{value}`")
    lines.extend(
        [
            "",
            "## 标签重建要求",
            "",
            "现有单试件窗口采用两套预测时域。联合 LOSO 不直接混用旧标签；下一步在独立缓存中按 200/1000/5000 ms 重建标签和采样索引，不覆盖单试件结果。",
            "",
            "低速试件的摄像间隔为 200 ms，因此 200 ms 是跨全部试件可解释的最短公共时域。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    workspace = resolve(config_path.parent, config.get("workspace_root", "../.."))
    workspace = workspace.resolve()
    output_dir = resolve(workspace, config["output_root"])
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = load_json(resolve(workspace, config["required_scope_audit"]))
    if audit.get("status") != "pass":
        raise RuntimeError("The required 16-specimen scope audit has not passed")
    manifest_rows, excluded_ids = read_scope(workspace, config)
    specimen_rows, configured_horizons = build_specimen_index(
        workspace, manifest_rows, config["target"]["primary_event_id"]
    )
    maximum_training_samples = int(config["sampling"]["maximum_training_samples_per_fold"])
    fold_csv_rows, folds = build_folds(specimen_rows, maximum_training_samples)
    specimen_ids = [row["specimen_id"] for row in specimen_rows]
    common_horizons = [int(value) for value in config["target"]["prediction_horizons_ms"]]
    slowest_frame_interval_ms = max(float(row["camera_frame_interval_ms"]) for row in specimen_rows)

    leakage_checks = {
        "exactly_one_test_specimen_per_fold": all(len({fold["test_specimen_id"]}) == 1 for fold in folds),
        "test_specimen_absent_from_training": all(
            fold["test_specimen_id"] not in fold["train_specimen_ids"] for fold in folds
        ),
        "all_other_specimens_present_in_training": all(
            len(fold["train_specimen_ids"]) == len(specimen_ids) - 1 for fold in folds
        ),
        "every_specimen_is_tested_once": Counter(
            fold["test_specimen_id"] for fold in folds
        ) == Counter(specimen_ids),
        "excluded_specimens_absent": not (set(excluded_ids) & set(specimen_ids)),
        "scope_audit_passed": audit.get("status") == "pass",
        "all_window_validations_passed": all(
            row["window_validation_status"] == "pass" for row in specimen_rows
        ),
        "all_ground_truth_confirmed": all(
            row["ground_truth_status"] == "confirmed_by_user" for row in specimen_rows
        ),
        "common_minimum_horizon_respects_slowest_camera": min(common_horizons)
        >= math.ceil(slowest_frame_interval_ms - 1.0e-9),
    }
    raw_training_counts = [fold["raw_training_unique_windows"] for fold in folds]
    single_kernel_mib = 4 * maximum_training_samples**2 / 1024**2
    common_cache_report_path = output_dir / "common_horizon_build_report.json"
    common_cache_current = False
    if common_cache_report_path.exists():
        common_cache_report = load_json(common_cache_report_path)
        common_cache_current = bool(
            common_cache_report.get("status") == "pass"
            and common_cache_report.get("specimen_count") == len(specimen_rows)
            and common_cache_report.get("common_prediction_horizons_ms") == common_horizons
            and common_cache_report.get("primary_event_id")
            == config["target"]["primary_event_id"]
        )
    report = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(leakage_checks.values()) else "fail",
        "study_name": config["study_name"],
        "counts": {
            "specimens": len(specimen_rows),
            "folds": len(folds),
            "rate_groups": dict(Counter(row["rate_group"] for row in specimen_rows)),
            "total_available_unique_windows_before_fold_sampling": sum(
                int(row["available_unique_windows"]) for row in specimen_rows
            ),
        },
        "excluded_specimens": excluded_ids,
        "current_per_specimen_prediction_horizons_ms": configured_horizons,
        "common_prediction_horizons_ms": common_horizons,
        "slowest_camera_frame_interval_ms": slowest_frame_interval_ms,
        "source_specimen_horizons_are_heterogeneous": len(
            {tuple(value) for value in configured_horizons.values()}
        )
        > 1,
        "common_label_cache_status": "current" if common_cache_current else "rebuild_required",
        "label_rebuild_required": not common_cache_current,
        "label_rebuild_location": str(output_dir / "common_horizon_windows"),
        "leakage_checks": leakage_checks,
        "resource_plan": {
            "compute_profile": "laptop_safe",
            "dtype": config["kernel"]["dtype"],
            "maximum_training_samples_per_fold": maximum_training_samples,
            "minimum_raw_training_windows": min(raw_training_counts),
            "maximum_raw_training_windows": max(raw_training_counts),
            "single_kernel_mib_at_cap": single_kernel_mib,
            "five_base_kernels_mib_at_cap": 5 * single_kernel_mib,
            "parallel_outer_folds": config["optimization"]["parallel_outer_folds"],
        },
        "folds": folds,
    }
    write_csv(output_dir / "study_specimen_index.csv", specimen_rows)
    write_csv(output_dir / "loso_folds.csv", fold_csv_rows)
    (output_dir / "loso_folds.json").write_text(
        json.dumps({"schema_version": "1.0", "folds": folds}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "loso_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(report, output_dir / "loso_readiness_report.md")
    print(json.dumps({key: value for key, value in report.items() if key not in {"folds", "current_per_specimen_prediction_horizons_ms"}}, ensure_ascii=False, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
