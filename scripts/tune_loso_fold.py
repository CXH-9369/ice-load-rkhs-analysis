#!/usr/bin/env python3
"""Training-only grouped inner selection for one outer LOSO fold."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path
from typing import Any

import project_dependencies  # noqa: F401
import numpy as np

from kernel_logistic_regression import (
    fit_kernel_logistic_regression,
    rate_balanced_weights,
    specimen_balanced_weights,
)
from rkhs_kernels import DTYPE, composite_kernel, fit_robust_standardizer, transform_and_flatten_sequence_features
from run_loso_fold import (
    SelectedSpecimen,
    binary_auc,
    concatenate_selected,
    deterministic_event_aware_sample,
    fit_pressure_rms_per_sample,
    fit_rbf_train_cross,
    loading_rate_train_cross,
    metric_summary,
    prepare_pressure_vectors_per_sample,
    select_specimen,
    select_training_threshold,
    combine_pressure_kernels,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_inner_splits(
    specimens: list[SelectedSpecimen], fold_count: int
) -> list[dict[str, list[str]]]:
    if fold_count != 2:
        raise ValueError("The current laptop-safe grouped splitter requires exactly two folds")
    by_group: dict[str, list[str]] = {}
    for item in specimens:
        by_group.setdefault(item.rate_group, []).append(item.specimen_id)
    validation = [[] for _ in range(fold_count)]
    for group in ("low", "medium", "high"):
        ids = sorted(by_group.get(group, []))
        for index, specimen_id in enumerate(ids):
            validation[index % fold_count].append(specimen_id)
    all_ids = [item.specimen_id for item in specimens]
    result = []
    for index in range(fold_count):
        validation_ids = validation[index]
        training_ids = [value for value in all_ids if value not in validation_ids]
        if not training_ids or not validation_ids:
            raise ValueError("Every inner split must have training and validation specimens")
        result.append({"train": training_ids, "validation": validation_ids})
    return result


def concatenate_all(specimens: list[SelectedSpecimen]) -> dict[str, np.ndarray]:
    return concatenate_selected(
        specimens,
        {
            item.specimen_id: np.arange(len(item.labels), dtype=np.int32)
            for item in specimens
        },
    )


def build_base_kernels(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    chunk_size: int,
    seed: int,
    pressure_weight: float,
    gradient_weight: float,
    mechanics_feature_names: list[str],
    phase_feature_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    all_feature_names = train["mechanics_feature_names"].astype(str).tolist()
    mechanics_indices = [all_feature_names.index(name) for name in mechanics_feature_names]
    train_mechanics_values = train["mechanics_features"][:, :, mechanics_indices]
    validation_mechanics_values = validation["mechanics_features"][:, :, mechanics_indices]
    train_mechanics_mask = train["mechanics_valid_mask"][:, :, mechanics_indices]
    validation_mechanics_mask = validation["mechanics_valid_mask"][:, :, mechanics_indices]
    standardizer = fit_robust_standardizer(
        train_mechanics_values, train_mechanics_mask
    )
    train_mechanics = transform_and_flatten_sequence_features(
        train_mechanics_values, train_mechanics_mask, standardizer
    )
    validation_mechanics = transform_and_flatten_sequence_features(
        validation_mechanics_values, validation_mechanics_mask, standardizer
    )
    km_train, km_validation, mechanics_scale, mechanics_degenerate = fit_rbf_train_cross(
        train_mechanics, validation_mechanics, chunk_size, seed + 101
    )
    del train_mechanics, validation_mechanics
    gc.collect()

    pressure_rms = fit_pressure_rms_per_sample(
        train["pressure_channels"], train["footprint_overlap_fraction"]
    )
    train_pressure, train_gradient = prepare_pressure_vectors_per_sample(
        train["pressure_channels"], train["footprint_overlap_fraction"], pressure_rms
    )
    validation_pressure, validation_gradient = prepare_pressure_vectors_per_sample(
        validation["pressure_channels"], validation["footprint_overlap_fraction"], pressure_rms
    )
    kp_train, kp_validation, pressure_info = combine_pressure_kernels(
        train_pressure,
        validation_pressure,
        train_gradient,
        validation_gradient,
        pressure_weight,
        gradient_weight,
        chunk_size,
        seed,
    )
    del train_pressure, validation_pressure, train_gradient, validation_gradient
    gc.collect()

    phase_index = all_feature_names.index(phase_feature_name)
    train_phase_raw = train["mechanics_features"][:, -1, phase_index][:, None, None]
    validation_phase_raw = validation["mechanics_features"][:, -1, phase_index][:, None, None]
    train_phase_valid = train["mechanics_valid_mask"][:, -1, phase_index][:, None, None]
    validation_phase_valid = validation["mechanics_valid_mask"][:, -1, phase_index][:, None, None]
    phase_standardizer = fit_robust_standardizer(train_phase_raw, train_phase_valid)
    train_phase = transform_and_flatten_sequence_features(
        train_phase_raw, train_phase_valid, phase_standardizer
    )
    validation_phase = transform_and_flatten_sequence_features(
        validation_phase_raw, validation_phase_valid, phase_standardizer
    )
    kt_train, kt_validation, phase_scale, phase_degenerate = fit_rbf_train_cross(
        train_phase, validation_phase, chunk_size, seed + 211
    )
    kr_train, kr_validation, rate_scale, rate_degenerate = loading_rate_train_cross(
        np.asarray(train["log_loading_rate"], dtype=DTYPE),
        np.asarray(validation["log_loading_rate"], dtype=DTYPE),
        train["specimen_id"],
        chunk_size,
        seed + 307,
    )
    kernels = {
        "km_train": km_train,
        "km_validation": km_validation,
        "kp_train": kp_train,
        "kp_validation": kp_validation,
        "kt_train": kt_train,
        "kt_validation": kt_validation,
        "kr_train": kr_train,
        "kr_validation": kr_validation,
    }
    metadata = {
        "mechanics_length_scale": mechanics_scale,
        "mechanics_degenerate": mechanics_degenerate,
        "mechanics_feature_names": mechanics_feature_names,
        "pressure": pressure_info,
        "pressure_channel_rms": pressure_rms.astype(float).tolist(),
        "phase_length_scale": phase_scale,
        "phase_degenerate": phase_degenerate,
        "phase_feature_name": phase_feature_name,
        "rate_length_scale": rate_scale,
        "rate_degenerate": rate_degenerate,
    }
    return kernels, metadata


def save_cache(
    root: Path,
    kernels: dict[str, np.ndarray],
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, values in kernels.items():
        np.save(root / f"{name}.npy", values, allow_pickle=False)
    np.savez_compressed(
        root / "split_arrays.npz",
        train_labels=train["labels"],
        train_specimen_id=train["specimen_id"],
        train_rate_group=train["rate_group"],
        validation_labels=validation["labels"],
        validation_specimen_id=validation["specimen_id"],
        validation_rate_group=validation["rate_group"],
        validation_endpoint_time_s=validation["endpoint_time_s"],
    )
    (root / "kernel_fit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_cache(root: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    kernels = {
        name: np.load(root / f"{name}.npy", mmap_mode="r")
        for name in (
            "km_train",
            "km_validation",
            "kp_train",
            "kp_validation",
            "kt_train",
            "kt_validation",
            "kr_train",
            "kr_validation",
        )
    }
    with np.load(root / "split_arrays.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    return kernels, arrays


def compose(
    kernels: dict[str, np.ndarray], beta: list[float], rho: float
) -> tuple[np.ndarray, np.ndarray]:
    train = composite_kernel(
        kernels["km_train"],
        kernels["kp_train"],
        kernels["kt_train"],
        kernels["kr_train"],
        beta_mechanics=beta[0],
        beta_pressure=beta[1],
        beta_interaction=beta[2],
        rho=rho,
        unit_diagonal=True,
    )
    validation = composite_kernel(
        kernels["km_validation"],
        kernels["kp_validation"],
        kernels["kt_validation"],
        kernels["kr_validation"],
        beta_mechanics=beta[0],
        beta_pressure=beta[1],
        beta_interaction=beta[2],
        rho=rho,
        unit_diagonal=False,
    )
    return train, validation


def weighted_log_loss(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> float:
    target = np.asarray(labels, dtype=np.float64)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    sample_weight = np.asarray(weights, dtype=np.float64)
    sample_weight /= np.sum(sample_weight)
    return float(
        -np.sum(
            sample_weight
            * (target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))
        )
    )


def aggregate_score(
    labels: list[np.ndarray],
    probabilities: list[np.ndarray],
    specimen_ids: list[np.ndarray],
    rate_groups: list[np.ndarray],
    weighting: str,
) -> dict[str, Any]:
    target = np.concatenate(labels)
    probability = np.concatenate(probabilities)
    specimen = np.concatenate(specimen_ids)
    groups = np.concatenate(rate_groups)
    if weighting == "rate_balanced":
        weights = rate_balanced_weights(specimen, groups)
    elif weighting == "specimen_balanced":
        weights = specimen_balanced_weights(specimen)
    else:
        raise ValueError(weighting)
    return {
        "weighted_log_loss": weighted_log_loss(target, probability, weights),
        "window_auc": binary_auc(target, probability),
        "labels": target,
        "probabilities": probability,
        "specimen_ids": specimen,
        "rate_groups": groups,
        "weights": weights,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/loso/loso_v1.json"))
    parser.add_argument("--fold-id", default="fold_01")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_json(args.config)
    workspace = args.config.resolve().parents[2]
    output_root = workspace / config["output_root"]
    folds = load_json(output_root / "loso_folds.json")["folds"]
    outer_fold = next(item for item in folds if item["fold_id"] == args.fold_id)
    output_dir = args.output_dir or output_root / "fold_runs" / args.fold_id / "inner_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "kernel_cache"
    common_root = output_root / "common_horizon_windows"
    inner_config = config["optimization"]["inner_selection"]
    event_id = config["target"]["primary_event_id"]
    horizon_ms = int(config["target"]["primary_prediction_horizon_ms"])
    seed = int(config["validation"]["random_seed"])
    chunk_size = int(config["kernel"]["chunk_size"])
    feature_policy = config["kernel"].get(
        "feature_policy",
        {
            "mechanics_feature_names": [
                "machine_force_kn",
                "machine_force_backward_rate_kn_per_s",
                "machine_deformation_from_first_record_mm",
                "machine_engineering_stress_actual_mpa",
                "machine_engineering_strain_actual_from_first_record_pct",
            ],
            "phase_feature_name": "machine_engineering_strain_actual_from_first_record_pct",
        },
    )

    rate_by_id: dict[str, str] = {}
    with (output_root / "study_specimen_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rate_by_id[row["specimen_id"]] = row["rate_group"]
    outer_train = [
        select_specimen(common_root, specimen_id, rate_by_id[specimen_id], event_id, horizon_ms)
        for specimen_id in outer_fold["train_specimen_ids"]
    ]
    by_id = {item.specimen_id: item for item in outer_train}
    inner_splits = make_inner_splits(outer_train, int(inner_config["fold_count"]))
    split_summaries = []
    candidates = [
        {"beta": [float(value) for value in beta], "rho": float(rho)}
        for beta in config["kernel"]["beta_candidates"]
        for rho in config["kernel"]["rho_candidates"]
    ]
    stage1: dict[int, dict[str, Any]] = {
        index: {
            "labels": [],
            "probabilities": [],
            "specimen_ids": [],
            "rate_groups": [],
            "optimizer_success": [],
        }
        for index in range(len(candidates))
    }
    for split_number, split in enumerate(inner_splits, start=1):
        print(f"Building inner split {split_number}/{len(inner_splits)}", flush=True)
        train_specimens = [by_id[value] for value in split["train"]]
        validation_specimens = [by_id[value] for value in split["validation"]]
        selected = deterministic_event_aware_sample(
            train_specimens,
            int(inner_config["maximum_training_samples_per_inner_fold"]),
            seed + split_number * 10000,
        )
        train = concatenate_selected(train_specimens, selected)
        validation = concatenate_all(validation_specimens)
        kernels, kernel_metadata = build_base_kernels(
            train,
            validation,
            chunk_size,
            seed + split_number * 1000,
            float(config["kernel"]["pilot_hyperparameters"]["pressure_weight"]),
            float(config["kernel"]["pilot_hyperparameters"]["gradient_weight"]),
            list(feature_policy["mechanics_feature_names"]),
            str(feature_policy["phase_feature_name"]),
        )
        cache_dir = cache_root / f"inner_{split_number:02d}"
        save_cache(cache_dir, kernels, train, validation, kernel_metadata)
        split_summaries.append(
            {
                "inner_fold": split_number,
                "training_specimen_ids": split["train"],
                "validation_specimen_ids": split["validation"],
                "training_samples_before_cap": int(sum(len(item.labels) for item in train_specimens)),
                "training_samples_selected": int(len(train["labels"])),
                "validation_samples": int(len(validation["labels"])),
                "kernel_fit": kernel_metadata,
            }
        )
        sample_weight = rate_balanced_weights(train["specimen_id"], train["rate_group"])
        for candidate_index, candidate in enumerate(candidates):
            print(
                f"  kernel candidate {candidate_index + 1:02d}/{len(candidates):02d}",
                flush=True,
            )
            kernel_train, kernel_validation = compose(
                kernels, candidate["beta"], candidate["rho"]
            )
            model = fit_kernel_logistic_regression(
                kernel_train,
                train["labels"],
                sample_weight=sample_weight,
                regularization=float(inner_config["kernel_selection_regularization"]),
                max_iterations=int(inner_config["maximum_iterations"]),
                gradient_tolerance=float(inner_config["gradient_tolerance"]),
            )
            stage1[candidate_index]["labels"].append(validation["labels"])
            stage1[candidate_index]["probabilities"].append(
                model.predict_proba(kernel_validation)
            )
            stage1[candidate_index]["specimen_ids"].append(validation["specimen_id"])
            stage1[candidate_index]["rate_groups"].append(validation["rate_group"])
            stage1[candidate_index]["optimizer_success"].append(model.optimizer_success)
            del kernel_train, kernel_validation, model
        del kernels, train, validation
        gc.collect()

    kernel_rows = []
    for candidate_index, candidate in enumerate(candidates):
        values = stage1[candidate_index]
        score = aggregate_score(
            values["labels"],
            values["probabilities"],
            values["specimen_ids"],
            values["rate_groups"],
            "rate_balanced",
        )
        kernel_rows.append(
            {
                "candidate_index": candidate_index,
                "beta_mechanics": candidate["beta"][0],
                "beta_pressure": candidate["beta"][1],
                "beta_interaction": candidate["beta"][2],
                "rho": candidate["rho"],
                "oof_rate_balanced_log_loss": score["weighted_log_loss"],
                "oof_window_auc": score["window_auc"],
                "all_inner_optimizers_converged": all(values["optimizer_success"]),
            }
        )
    kernel_rows.sort(key=lambda row: (row["oof_rate_balanced_log_loss"], -float(row["oof_window_auc"] or 0.0)))
    best_kernel = kernel_rows[0]
    best_beta = [
        float(best_kernel["beta_mechanics"]),
        float(best_kernel["beta_pressure"]),
        float(best_kernel["beta_interaction"]),
    ]
    best_rho = float(best_kernel["rho"])
    with (output_dir / "kernel_candidate_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kernel_rows[0]))
        writer.writeheader()
        writer.writerows(kernel_rows)

    regularization_rows = []
    regularization_oof: dict[float, dict[str, list[np.ndarray]]] = {}
    for regularization in map(float, config["kernel"]["regularization_candidates"]):
        accumulated = {
            "labels": [],
            "probabilities": [],
            "specimen_ids": [],
            "rate_groups": [],
            "optimizer_success": [],
        }
        print(f"Regularization candidate {regularization:g}", flush=True)
        for split_number in range(1, len(inner_splits) + 1):
            kernels, arrays = load_cache(cache_root / f"inner_{split_number:02d}")
            kernel_train, kernel_validation = compose(kernels, best_beta, best_rho)
            sample_weight = rate_balanced_weights(
                arrays["train_specimen_id"], arrays["train_rate_group"]
            )
            model = fit_kernel_logistic_regression(
                kernel_train,
                arrays["train_labels"],
                sample_weight=sample_weight,
                regularization=regularization,
                max_iterations=int(inner_config["refinement_maximum_iterations"]),
                gradient_tolerance=float(inner_config["refinement_gradient_tolerance"]),
            )
            accumulated["labels"].append(arrays["validation_labels"])
            accumulated["probabilities"].append(model.predict_proba(kernel_validation))
            accumulated["specimen_ids"].append(arrays["validation_specimen_id"])
            accumulated["rate_groups"].append(arrays["validation_rate_group"])
            accumulated["optimizer_success"].append(model.optimizer_success)
            del kernels, arrays, kernel_train, kernel_validation, model
            gc.collect()
        score = aggregate_score(
            accumulated["labels"],
            accumulated["probabilities"],
            accumulated["specimen_ids"],
            accumulated["rate_groups"],
            "rate_balanced",
        )
        regularization_rows.append(
            {
                "regularization": regularization,
                "oof_rate_balanced_log_loss": score["weighted_log_loss"],
                "oof_window_auc": score["window_auc"],
                "all_inner_optimizers_converged": all(accumulated["optimizer_success"]),
            }
        )
        regularization_oof[regularization] = accumulated
    regularization_rows.sort(
        key=lambda row: (row["oof_rate_balanced_log_loss"], -float(row["oof_window_auc"] or 0.0))
    )
    best_regularization = float(regularization_rows[0]["regularization"])
    with (output_dir / "regularization_candidate_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(regularization_rows[0]))
        writer.writeheader()
        writer.writerows(regularization_rows)

    threshold_results: dict[str, Any] = {}
    for weighting in ("specimen_balanced", "rate_balanced"):
        accumulated = {
            "labels": [],
            "probabilities": [],
            "specimen_ids": [],
            "rate_groups": [],
            "optimizer_success": [],
        }
        for split_number in range(1, len(inner_splits) + 1):
            kernels, arrays = load_cache(cache_root / f"inner_{split_number:02d}")
            kernel_train, kernel_validation = compose(kernels, best_beta, best_rho)
            if weighting == "rate_balanced":
                sample_weight = rate_balanced_weights(
                    arrays["train_specimen_id"], arrays["train_rate_group"]
                )
            else:
                sample_weight = specimen_balanced_weights(arrays["train_specimen_id"])
            model = fit_kernel_logistic_regression(
                kernel_train,
                arrays["train_labels"],
                sample_weight=sample_weight,
                regularization=best_regularization,
                max_iterations=int(inner_config["refinement_maximum_iterations"]),
                gradient_tolerance=float(inner_config["refinement_gradient_tolerance"]),
            )
            accumulated["labels"].append(arrays["validation_labels"])
            accumulated["probabilities"].append(model.predict_proba(kernel_validation))
            accumulated["specimen_ids"].append(arrays["validation_specimen_id"])
            accumulated["rate_groups"].append(arrays["validation_rate_group"])
            accumulated["optimizer_success"].append(model.optimizer_success)
            del kernels, arrays, kernel_train, kernel_validation, model
            gc.collect()
        score = aggregate_score(
            accumulated["labels"],
            accumulated["probabilities"],
            accumulated["specimen_ids"],
            accumulated["rate_groups"],
            weighting,
        )
        threshold, threshold_score = select_training_threshold(
            score["labels"], score["probabilities"], score["weights"]
        )
        threshold_results[weighting] = {
            "threshold": threshold,
            "weighted_balanced_accuracy": threshold_score,
            "weighted_log_loss": score["weighted_log_loss"],
            "window_auc": score["window_auc"],
            "all_inner_optimizers_converged": all(accumulated["optimizer_success"]),
            "unweighted_oof_metrics": metric_summary(
                score["labels"], score["probabilities"], threshold
            ),
        }

    selection_converged = bool(
        best_kernel["all_inner_optimizers_converged"]
        and regularization_rows[0]["all_inner_optimizers_converged"]
        and all(
            value["all_inner_optimizers_converged"]
            for value in threshold_results.values()
        )
    )
    result = {
        "schema_version": "1.0",
        "status": "pass" if selection_converged else "optimizer_warning",
        "scope": "outer-training-only grouped inner selection",
        "outer_fold_id": args.fold_id,
        "outer_held_out_specimen_id": outer_fold["test_specimen_id"],
        "outer_held_out_used_for_selection": False,
        "target": {"event_id": event_id, "prediction_horizon_ms": horizon_ms},
        "inner_selection_config": inner_config,
        "inner_splits": split_summaries,
        "selected_hyperparameters": {
            "beta": best_beta,
            "rho": best_rho,
            "regularization": best_regularization,
            "pressure_weight": float(config["kernel"]["pilot_hyperparameters"]["pressure_weight"]),
            "gradient_weight": float(config["kernel"]["pilot_hyperparameters"]["gradient_weight"]),
            "mechanics_feature_names": list(feature_policy["mechanics_feature_names"]),
            "phase_feature_name": str(feature_policy["phase_feature_name"]),
        },
        "best_kernel_candidate": best_kernel,
        "best_regularization_candidate": regularization_rows[0],
        "thresholds": threshold_results,
        "candidate_counts": {
            "kernel_combinations": len(candidates),
            "regularization_values": len(regularization_rows),
        },
        "outputs": {
            "kernel_candidate_scores_csv": str(output_dir / "kernel_candidate_scores.csv"),
            "regularization_candidate_scores_csv": str(
                output_dir / "regularization_candidate_scores.csv"
            ),
            "kernel_cache": str(cache_root),
        },
        "total_seconds": float(time.perf_counter() - started),
    }
    output_json = output_dir / "inner_selection.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "selected_hyperparameters": result["selected_hyperparameters"],
        "thresholds": result["thresholds"],
        "output": str(output_json),
        "total_seconds": result["total_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
