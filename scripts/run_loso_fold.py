#!/usr/bin/env python3
"""Run one leakage-safe LOSO RKHS/KLR fold on common-horizon caches."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from kernel_logistic_regression import (
    fit_kernel_logistic_regression,
    rate_balanced_weights,
    specimen_balanced_weights,
)
from rkhs_kernels import (
    DTYPE,
    RobustStandardizer,
    composite_kernel,
    fit_robust_standardizer,
    median_length_scale_from_squared_distance,
    rbf_from_squared_distance,
    squared_euclidean_blockwise,
    transform_and_flatten_sequence_features,
)


@dataclass
class SelectedSpecimen:
    specimen_id: str
    rate_group: str
    event_interval_s: tuple[float, float]
    tensor_indices: np.ndarray
    labels: np.ndarray
    arrays: dict[str, np.ndarray]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def interval_label(
    endpoint_s: float, event_start_s: float, event_end_s: float, horizon_s: float
) -> int | None:
    horizon_end = endpoint_s + horizon_s
    if horizon_end < event_start_s:
        return 0
    if horizon_end >= event_end_s:
        return 1
    return None


def select_specimen(
    common_root: Path,
    specimen_id: str,
    rate_group: str,
    event_id: str,
    horizon_ms: int,
) -> SelectedSpecimen:
    root = common_root / specimen_id
    metadata = load_json(root / "event_window_metadata.json")
    if str(metadata["specimen_id"]) != specimen_id:
        raise ValueError(f"Metadata specimen mismatch for {specimen_id}")
    event = metadata["target_definition"]["events"][event_id]
    event_start, event_end = map(float, event["target_time_interval_s"])
    with np.load(root / "rkhs_causal_windows.npz", allow_pickle=False) as source:
        endpoint_all = np.asarray(source["endpoint_time_s"], dtype=np.float64)
        eligible = np.flatnonzero(endpoint_all < event_start)
        labels_raw = [
            interval_label(
                float(endpoint_all[index]), event_start, event_end, horizon_ms / 1000.0
            )
            for index in eligible
        ]
        definite = np.asarray([value is not None for value in labels_raw], dtype=bool)
        selected = eligible[definite]
        labels = np.asarray(
            [value for value in labels_raw if value is not None], dtype=np.int32
        )
        wanted = (
            "pressure_channels",
            "mechanics_features",
            "mechanics_valid_mask",
            "mechanics_feature_names",
            "endpoint_pressure_frame",
            "endpoint_time_s",
            "representative_loading_rate_per_s",
            "log_loading_rate",
            "footprint_overlap_fraction",
        )
        arrays: dict[str, np.ndarray] = {}
        for key in wanted:
            values = source[key]
            if key in {"mechanics_feature_names", "footprint_overlap_fraction"}:
                arrays[key] = np.asarray(values)
            else:
                arrays[key] = np.asarray(values[selected])
    if len(selected) == 0:
        raise ValueError(f"No eligible definite windows for {specimen_id}")
    return SelectedSpecimen(
        specimen_id=specimen_id,
        rate_group=rate_group,
        event_interval_s=(event_start, event_end),
        tensor_indices=selected.astype(np.int32),
        labels=labels,
        arrays=arrays,
    )


def deterministic_event_aware_sample(
    specimens: list[SelectedSpecimen], maximum: int, seed: int
) -> dict[str, np.ndarray]:
    """Keep all positives and distribute remaining negative capacity across specimens."""
    total = sum(len(item.labels) for item in specimens)
    if total <= maximum:
        return {
            item.specimen_id: np.arange(len(item.labels), dtype=np.int32)
            for item in specimens
        }
    positives = {
        item.specimen_id: np.flatnonzero(item.labels == 1).astype(np.int32)
        for item in specimens
    }
    positive_total = sum(len(values) for values in positives.values())
    if positive_total > maximum:
        raise ValueError(
            f"Positive windows ({positive_total}) exceed training cap ({maximum})"
        )
    negative_capacity = maximum - positive_total
    negative_counts = {
        item.specimen_id: int(np.count_nonzero(item.labels == 0)) for item in specimens
    }
    allocations = {item.specimen_id: 0 for item in specimens}
    active = [item.specimen_id for item in specimens if negative_counts[item.specimen_id]]
    while negative_capacity > 0 and active:
        progressed = False
        for specimen_id in active:
            if allocations[specimen_id] < negative_counts[specimen_id]:
                allocations[specimen_id] += 1
                negative_capacity -= 1
                progressed = True
                if negative_capacity == 0:
                    break
        if not progressed:
            break
        active = [
            specimen_id
            for specimen_id in active
            if allocations[specimen_id] < negative_counts[specimen_id]
        ]
    result: dict[str, np.ndarray] = {}
    for ordinal, item in enumerate(specimens):
        negatives = np.flatnonzero(item.labels == 0).astype(np.int32)
        take = allocations[item.specimen_id]
        if take < len(negatives):
            rng = np.random.default_rng(seed + 1009 * (ordinal + 1))
            negatives = np.sort(rng.choice(negatives, size=take, replace=False))
        chosen = np.sort(np.concatenate([positives[item.specimen_id], negatives]))
        result[item.specimen_id] = chosen.astype(np.int32)
    if sum(len(values) for values in result.values()) != maximum:
        raise RuntimeError("Sampling failed to satisfy the requested training cap")
    return result


def concatenate_selected(
    specimens: list[SelectedSpecimen], local_indices: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    feature_names = [tuple(item.arrays["mechanics_feature_names"].astype(str)) for item in specimens]
    if len(set(feature_names)) != 1:
        raise ValueError("Mechanics feature order differs between specimens")
    output: dict[str, np.ndarray] = {
        "mechanics_feature_names": np.asarray(feature_names[0]),
    }
    for key in (
        "pressure_channels",
        "mechanics_features",
        "mechanics_valid_mask",
        "endpoint_pressure_frame",
        "endpoint_time_s",
        "representative_loading_rate_per_s",
        "log_loading_rate",
    ):
        output[key] = np.concatenate(
            [item.arrays[key][local_indices[item.specimen_id]] for item in specimens], axis=0
        )
    output["labels"] = np.concatenate(
        [item.labels[local_indices[item.specimen_id]] for item in specimens]
    ).astype(np.int32)
    output["tensor_indices"] = np.concatenate(
        [item.tensor_indices[local_indices[item.specimen_id]] for item in specimens]
    ).astype(np.int32)
    output["specimen_id"] = np.concatenate(
        [
            np.full(len(local_indices[item.specimen_id]), item.specimen_id)
            for item in specimens
        ]
    )
    output["rate_group"] = np.concatenate(
        [
            np.full(len(local_indices[item.specimen_id]), item.rate_group)
            for item in specimens
        ]
    )
    output["footprint_overlap_fraction"] = np.concatenate(
        [
            np.repeat(
                item.arrays["footprint_overlap_fraction"][None, :, :],
                len(local_indices[item.specimen_id]),
                axis=0,
            )
            for item in specimens
        ],
        axis=0,
    ).astype(DTYPE, copy=False)
    return output


def fit_pressure_rms_per_sample(
    fields: np.ndarray, weights: np.ndarray, chunk_size: int = 64
) -> np.ndarray:
    values = np.asarray(fields, dtype=DTYPE)
    masks = np.asarray(weights, dtype=DTYPE)
    if values.ndim != 5 or values.shape[2] != 3:
        raise ValueError("pressure fields must have shape (N,H,3,Y,X)")
    if masks.shape != (len(values), *values.shape[-2:]):
        raise ValueError("per-sample footprint weights do not match pressure fields")
    pressure_ss = 0.0
    gradient_ss = 0.0
    weighted_cells = 0.0
    for start in range(0, len(values), chunk_size):
        block = values[start : start + chunk_size]
        block_weights = masks[start : start + chunk_size]
        pressure_ss += float(
            np.einsum(
                "nhyx,nyx,nhyx->",
                block[:, :, 0],
                block_weights,
                block[:, :, 0],
                dtype=np.float64,
                optimize=True,
            )
        )
        gradient_ss += float(
            np.einsum(
                "nhcyx,nyx,nhcyx->",
                block[:, :, 1:],
                block_weights,
                block[:, :, 1:],
                dtype=np.float64,
                optimize=True,
            )
        )
        weighted_cells += float(np.sum(block_weights, dtype=np.float64))
    pressure_denominator = values.shape[1] * weighted_cells
    gradient_denominator = 2.0 * pressure_denominator
    if pressure_denominator <= 0:
        raise ValueError("pressure footprint weights have zero total area")
    result = np.asarray(
        [
            math.sqrt(pressure_ss / pressure_denominator),
            math.sqrt(gradient_ss / gradient_denominator),
        ],
        dtype=DTYPE,
    )
    result[~np.isfinite(result) | (result <= 1.0e-8)] = DTYPE(1.0)
    return result


def prepare_pressure_vectors_per_sample(
    fields: np.ndarray, weights: np.ndarray, channel_rms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fields, dtype=DTYPE)
    masks = np.asarray(weights, dtype=DTYPE)
    rms = np.asarray(channel_rms, dtype=DTYPE)
    history = values.shape[1]
    sums = np.sum(masks, axis=(1, 2), dtype=np.float64)
    if np.any(sums <= 0):
        raise ValueError("Every sample must have a positive footprint area")
    sqrt_weight = np.sqrt(masks).astype(DTYPE, copy=False)
    pressure_norm = (1.0 / np.sqrt(history * sums)).astype(DTYPE)
    gradient_norm = (1.0 / np.sqrt(2.0 * history * sums)).astype(DTYPE)
    pressure = (
        values[:, :, 0]
        * sqrt_weight[:, None, :, :]
        / rms[0]
        * pressure_norm[:, None, None, None]
    ).reshape(len(values), -1)
    gradient = (
        values[:, :, 1:]
        * sqrt_weight[:, None, None, :, :]
        / rms[1]
        * gradient_norm[:, None, None, None, None]
    ).reshape(len(values), -1)
    return pressure.astype(DTYPE, copy=False), gradient.astype(DTYPE, copy=False)


def fit_rbf_train_cross(
    train_vectors: np.ndarray,
    test_vectors: np.ndarray,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    train_distance = squared_euclidean_blockwise(train_vectors, chunk_size=chunk_size)
    length_scale, degenerate = median_length_scale_from_squared_distance(
        train_distance, seed=seed
    )
    if degenerate:
        train_kernel = np.ones_like(train_distance, dtype=DTYPE)
    else:
        train_kernel = rbf_from_squared_distance(train_distance, length_scale, copy=False)
    cross_distance = squared_euclidean_blockwise(
        test_vectors, train_vectors, chunk_size=chunk_size
    )
    if degenerate:
        cross_kernel = np.ones_like(cross_distance, dtype=DTYPE)
    else:
        cross_kernel = rbf_from_squared_distance(
            cross_distance, length_scale, copy=False, unit_diagonal=False
        )
    return train_kernel, cross_kernel, float(length_scale), bool(degenerate)


def combine_pressure_kernels(
    train_pressure: np.ndarray,
    test_pressure: np.ndarray,
    train_gradient: np.ndarray,
    test_gradient: np.ndarray,
    pressure_weight: float,
    gradient_weight: float,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    total = pressure_weight + gradient_weight
    pressure_weight /= total
    gradient_weight /= total
    p_train_distance = squared_euclidean_blockwise(train_pressure, chunk_size=chunk_size)
    p_scale, p_degenerate = median_length_scale_from_squared_distance(
        p_train_distance, seed=seed + 11
    )
    exponent_train = np.zeros_like(p_train_distance, dtype=DTYPE)
    if not p_degenerate and pressure_weight:
        exponent_train += DTYPE(pressure_weight / (p_scale * p_scale)) * p_train_distance
    del p_train_distance
    gc.collect()
    g_train_distance = squared_euclidean_blockwise(train_gradient, chunk_size=chunk_size)
    g_scale, g_degenerate = median_length_scale_from_squared_distance(
        g_train_distance, seed=seed + 17
    )
    if not g_degenerate and gradient_weight:
        exponent_train += DTYPE(gradient_weight / (g_scale * g_scale)) * g_train_distance
    del g_train_distance
    exponent_train *= DTYPE(-0.5)
    np.exp(exponent_train, out=exponent_train)
    np.fill_diagonal(exponent_train, DTYPE(1.0))

    p_cross_distance = squared_euclidean_blockwise(
        test_pressure, train_pressure, chunk_size=chunk_size
    )
    exponent_cross = np.zeros_like(p_cross_distance, dtype=DTYPE)
    if not p_degenerate and pressure_weight:
        exponent_cross += DTYPE(pressure_weight / (p_scale * p_scale)) * p_cross_distance
    del p_cross_distance
    gc.collect()
    g_cross_distance = squared_euclidean_blockwise(
        test_gradient, train_gradient, chunk_size=chunk_size
    )
    if not g_degenerate and gradient_weight:
        exponent_cross += DTYPE(gradient_weight / (g_scale * g_scale)) * g_cross_distance
    del g_cross_distance
    exponent_cross *= DTYPE(-0.5)
    np.exp(exponent_cross, out=exponent_cross)
    return exponent_train, exponent_cross, {
        "pressure_length_scale": float(p_scale),
        "gradient_length_scale": float(g_scale),
        "pressure_degenerate": bool(p_degenerate),
        "gradient_degenerate": bool(g_degenerate),
        "pressure_weight": float(pressure_weight),
        "gradient_weight": float(gradient_weight),
    }


def loading_rate_train_cross(
    train_rates: np.ndarray,
    test_rates: np.ndarray,
    train_specimens: np.ndarray,
    chunk_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    unique_rates = []
    for specimen_id in np.unique(train_specimens.astype(str)):
        values = train_rates[train_specimens.astype(str) == specimen_id]
        if not np.allclose(values, values[0], rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(f"Inconsistent loading rates in {specimen_id}")
        unique_rates.append(float(values[0]))
    unique_distance = squared_euclidean_blockwise(
        np.asarray(unique_rates, dtype=DTYPE).reshape(-1, 1), chunk_size=chunk_size
    )
    scale, degenerate = median_length_scale_from_squared_distance(unique_distance, seed=seed)
    del unique_distance
    train_distance = squared_euclidean_blockwise(
        train_rates.reshape(-1, 1), chunk_size=chunk_size
    )
    cross_distance = squared_euclidean_blockwise(
        test_rates.reshape(-1, 1), train_rates.reshape(-1, 1), chunk_size=chunk_size
    )
    if degenerate:
        return (
            np.ones_like(train_distance, dtype=DTYPE),
            np.ones_like(cross_distance, dtype=DTYPE),
            float(scale),
            True,
        )
    return (
        rbf_from_squared_distance(train_distance, scale, copy=False),
        rbf_from_squared_distance(
            cross_distance, scale, copy=False, unit_diagonal=False
        ),
        float(scale),
        False,
    )


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    if not len(positive) or not len(negative):
        return None
    order = np.argsort(np.concatenate([negative, positive]), kind="mergesort")
    sorted_scores = np.concatenate([negative, positive])[order]
    sorted_labels = np.concatenate(
        [np.zeros(len(negative), dtype=np.int8), np.ones(len(positive), dtype=np.int8)]
    )[order]
    ranks = np.empty(len(order), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[start:stop] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[sorted_labels == 1]))
    return (positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (
        len(positive) * len(negative)
    )


def balanced_accuracy(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    weights: np.ndarray | None = None,
) -> tuple[float | None, float | None, float | None]:
    target = np.asarray(labels, dtype=np.int32)
    predicted = np.asarray(probabilities) >= threshold
    sample_weight = (
        np.ones(len(target), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    sensitivity = None
    specificity = None
    if np.any(target == 1):
        sensitivity = float(
            np.sum(sample_weight[(target == 1) & predicted]) / np.sum(sample_weight[target == 1])
        )
    if np.any(target == 0):
        specificity = float(
            np.sum(sample_weight[(target == 0) & ~predicted]) / np.sum(sample_weight[target == 0])
        )
    score = None if sensitivity is None or specificity is None else 0.5 * (sensitivity + specificity)
    return score, sensitivity, specificity


def select_training_threshold(
    labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    candidates = np.unique(
        np.concatenate(
            ([0.0], np.quantile(probabilities, np.linspace(0.0, 1.0, 1001)), [1.0])
        )
    )
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidates:
        score, _, _ = balanced_accuracy(labels, probabilities, float(threshold), weights)
        if score is not None and (
            score > best_score + 1.0e-12
            or (abs(score - best_score) <= 1.0e-12 and abs(threshold - 0.5) < abs(best_threshold - 0.5))
        ):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, float(best_score)


def metric_summary(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    target = np.asarray(labels, dtype=np.int32)
    probability = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    score, sensitivity, specificity = balanced_accuracy(target, probability, threshold)
    predicted = probability >= threshold
    return {
        "window_count": int(len(target)),
        "positive_count": int(np.count_nonzero(target == 1)),
        "negative_count": int(np.count_nonzero(target == 0)),
        "auc": binary_auc(target, probability),
        "balanced_accuracy": score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": float(np.mean(predicted == target)),
        "brier_score": float(np.mean((probability - target) ** 2)),
        "log_loss": float(
            -np.mean(target * np.log(probability) + (1 - target) * np.log(1 - probability))
        ),
        "predicted_positive_count": int(np.count_nonzero(predicted)),
    }


def model_summary(
    model: Any,
    threshold: float,
    threshold_score: float,
    threshold_source: str,
) -> dict[str, Any]:
    return {
        "optimizer_success": bool(model.optimizer_success),
        "optimizer_status": int(model.optimizer_status),
        "optimizer_message": str(model.optimizer_message),
        "iterations": int(model.iterations),
        "function_evaluations": int(model.function_evaluations),
        "initial_objective": float(model.initial_objective),
        "final_objective": float(model.final_objective),
        "objective_reduction": float(model.initial_objective - model.final_objective),
        "intercept": float(model.intercept),
        "alpha_l2_norm": float(np.linalg.norm(model.alpha)),
        "decision_threshold": float(threshold),
        "threshold_source": threshold_source,
        "selection_weighted_balanced_accuracy_at_threshold": float(threshold_score),
    }


def event_timing_summary(
    endpoint_times: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    horizon_s: float,
    event_interval_s: tuple[float, float],
) -> dict[str, Any]:
    order = np.argsort(endpoint_times)
    times = np.asarray(endpoint_times)[order]
    values = np.asarray(probabilities)[order]
    crossing = np.flatnonzero(values >= threshold)
    peak = int(np.argmax(values))
    center = 0.5 * (event_interval_s[0] + event_interval_s[1])
    crossing_prediction = None if not len(crossing) else float(times[crossing[0]] + horizon_s)
    peak_prediction = float(times[peak] + horizon_s)
    return {
        "event_interval_s": [float(event_interval_s[0]), float(event_interval_s[1])],
        "event_interval_center_s": float(center),
        "first_threshold_crossing_predicted_event_time_s": crossing_prediction,
        "first_threshold_crossing_error_to_interval_center_s": (
            None if crossing_prediction is None else float(crossing_prediction - center)
        ),
        "peak_probability_predicted_event_time_s": peak_prediction,
        "peak_probability_error_to_interval_center_s": float(peak_prediction - center),
        "maximum_probability": float(values[peak]),
    }


def write_training_manifest(path: Path, train: dict[str, np.ndarray]) -> None:
    fields = [
        "training_row_0_based",
        "specimen_id",
        "rate_group",
        "source_tensor_index_0_based",
        "endpoint_pressure_frame",
        "endpoint_time_s",
        "label",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(train["labels"])):
            writer.writerow(
                {
                    "training_row_0_based": index,
                    "specimen_id": train["specimen_id"][index],
                    "rate_group": train["rate_group"][index],
                    "source_tensor_index_0_based": int(train["tensor_indices"][index]),
                    "endpoint_pressure_frame": int(train["endpoint_pressure_frame"][index]),
                    "endpoint_time_s": float(train["endpoint_time_s"][index]),
                    "label": int(train["labels"][index]),
                }
            )


def plot_fold(
    path: Path,
    specimen_id: str,
    endpoint_times: np.ndarray,
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    event_interval_s: tuple[float, float],
    horizon_ms: int,
    histories: dict[str, np.ndarray],
) -> None:
    order = np.argsort(endpoint_times)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for name, values in probabilities.items():
        axes[0].plot(endpoint_times[order], values[order], linewidth=1.6, label=name)
        axes[0].axhline(thresholds[name], linestyle="--", linewidth=1.0, alpha=0.7)
    axes[0].scatter(
        endpoint_times[order], labels[order], s=9, c=np.where(labels[order] == 1, "#70ad47", "#c00000"),
        alpha=0.55, label=f"{horizon_ms} ms label", zorder=3,
    )
    axes[0].axvspan(event_interval_s[0], event_interval_s[1], color="#c00000", alpha=0.12)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("Held-out window endpoint time (s)")
    axes[0].set_ylabel("Predicted probability / label")
    axes[0].set_title(f"Held-out specimen: {specimen_id}")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    for name, history in histories.items():
        axes[1].plot(np.arange(len(history)), history, linewidth=1.5, label=name)
    axes[1].set_xlabel("L-BFGS callback step")
    axes[1].set_ylabel("Weighted objective")
    axes[1].set_title("Training-fold optimization convergence")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=9)
    fig.suptitle("Leakage-safe LOSO RKHS/KLR pilot fold", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/loso/loso_v1.json"))
    parser.add_argument("--fold-id", default="fold_01")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection-json", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_json(args.config)
    workspace = args.config.resolve().parents[2]
    output_root = workspace / config["output_root"]
    folds = load_json(output_root / "loso_folds.json")["folds"]
    fold = next((item for item in folds if item["fold_id"] == args.fold_id), None)
    if fold is None:
        raise KeyError(f"Unknown fold: {args.fold_id}")
    output_dir = args.output_dir or output_root / "fold_runs" / args.fold_id
    output_dir.mkdir(parents=True, exist_ok=True)
    common_root = output_root / "common_horizon_windows"
    event_id = str(config["target"]["primary_event_id"])
    horizon_ms = int(config["target"]["primary_prediction_horizon_ms"])
    seed = int(config["validation"]["random_seed"])
    chunk_size = int(config["kernel"]["chunk_size"])
    pilot = config["kernel"]["pilot_hyperparameters"]
    selection = None
    chosen = pilot
    if args.selection_json is not None:
        selection = load_json(args.selection_json)
        if selection.get("status") != "pass":
            raise ValueError("The supplied inner-selection result has not passed")
        if selection.get("outer_fold_id") != args.fold_id:
            raise ValueError("The inner-selection result belongs to a different outer fold")
        if selection.get("outer_held_out_used_for_selection") is not False:
            raise ValueError("The inner-selection result does not certify held-out isolation")
        chosen = selection["selected_hyperparameters"]
    all_mechanics_feature_names = None

    index_rows: dict[str, dict[str, str]] = {}
    with (output_root / "study_specimen_index.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            index_rows[row["specimen_id"]] = row
    train_specimens = [
        select_specimen(
            common_root, specimen_id, index_rows[specimen_id]["rate_group"], event_id, horizon_ms
        )
        for specimen_id in fold["train_specimen_ids"]
    ]
    test_id = str(fold["test_specimen_id"])
    test_specimen = select_specimen(
        common_root, test_id, index_rows[test_id]["rate_group"], event_id, horizon_ms
    )
    selected_local = deterministic_event_aware_sample(
        train_specimens, int(config["sampling"]["maximum_training_samples_per_fold"]), seed
    )
    train = concatenate_selected(train_specimens, selected_local)
    test = concatenate_selected(
        [test_specimen], {test_id: np.arange(len(test_specimen.labels), dtype=np.int32)}
    )
    if test_id in set(train["specimen_id"].astype(str)):
        raise RuntimeError("Held-out specimen leaked into training data")
    if len(np.unique(train["labels"])) < 2:
        raise ValueError("Training fold does not contain both target classes")
    write_training_manifest(output_dir / "training_sample_manifest.csv", train)
    kernel_started = time.perf_counter()

    all_mechanics_feature_names = train["mechanics_feature_names"].astype(str).tolist()
    mechanics_feature_names = chosen.get(
        "mechanics_feature_names", all_mechanics_feature_names
    )
    phase_feature_name = chosen.get(
        "phase_feature_name",
        "machine_engineering_strain_actual_from_first_record_pct",
    )
    mechanics_indices = [
        all_mechanics_feature_names.index(name) for name in mechanics_feature_names
    ]
    train_mechanics_values = train["mechanics_features"][:, :, mechanics_indices]
    test_mechanics_values = test["mechanics_features"][:, :, mechanics_indices]
    train_mechanics_mask = train["mechanics_valid_mask"][:, :, mechanics_indices]
    test_mechanics_mask = test["mechanics_valid_mask"][:, :, mechanics_indices]
    mechanics_standardizer = fit_robust_standardizer(
        train_mechanics_values, train_mechanics_mask
    )
    train_mechanics = transform_and_flatten_sequence_features(
        train_mechanics_values, train_mechanics_mask, mechanics_standardizer
    )
    test_mechanics = transform_and_flatten_sequence_features(
        test_mechanics_values, test_mechanics_mask, mechanics_standardizer
    )
    km_train, km_test, mechanics_scale, mechanics_degenerate = fit_rbf_train_cross(
        train_mechanics, test_mechanics, chunk_size, seed + 101
    )
    del train_mechanics, test_mechanics
    gc.collect()

    pressure_rms = fit_pressure_rms_per_sample(
        train["pressure_channels"], train["footprint_overlap_fraction"]
    )
    train_pressure, train_gradient = prepare_pressure_vectors_per_sample(
        train["pressure_channels"], train["footprint_overlap_fraction"], pressure_rms
    )
    test_pressure, test_gradient = prepare_pressure_vectors_per_sample(
        test["pressure_channels"], test["footprint_overlap_fraction"], pressure_rms
    )
    kp_train, kp_test, pressure_info = combine_pressure_kernels(
        train_pressure,
        test_pressure,
        train_gradient,
        test_gradient,
        float(chosen["pressure_weight"]),
        float(chosen["gradient_weight"]),
        chunk_size,
        seed,
    )
    del train_pressure, test_pressure, train_gradient, test_gradient
    del train["pressure_channels"], test["pressure_channels"]
    gc.collect()

    phase_index = all_mechanics_feature_names.index(phase_feature_name)
    train_phase_raw = train["mechanics_features"][:, -1, phase_index][:, None, None]
    test_phase_raw = test["mechanics_features"][:, -1, phase_index][:, None, None]
    train_phase_valid = train["mechanics_valid_mask"][:, -1, phase_index][:, None, None]
    test_phase_valid = test["mechanics_valid_mask"][:, -1, phase_index][:, None, None]
    phase_standardizer: RobustStandardizer = fit_robust_standardizer(
        train_phase_raw, train_phase_valid
    )
    train_phase = transform_and_flatten_sequence_features(
        train_phase_raw, train_phase_valid, phase_standardizer
    )
    test_phase = transform_and_flatten_sequence_features(
        test_phase_raw, test_phase_valid, phase_standardizer
    )
    kt_train, kt_test, phase_scale, phase_degenerate = fit_rbf_train_cross(
        train_phase, test_phase, chunk_size, seed + 211
    )
    del train_phase, test_phase
    gc.collect()

    kr_train, kr_test, rate_scale, rate_degenerate = loading_rate_train_cross(
        np.asarray(train["log_loading_rate"], dtype=DTYPE),
        np.asarray(test["log_loading_rate"], dtype=DTYPE),
        train["specimen_id"],
        chunk_size,
        seed + 307,
    )
    beta = [float(value) for value in chosen["beta"]]
    rho = float(chosen["rho"])
    composite_train = composite_kernel(
        km_train,
        kp_train,
        kt_train,
        kr_train,
        beta_mechanics=beta[0],
        beta_pressure=beta[1],
        beta_interaction=beta[2],
        rho=rho,
    )
    composite_test = composite_kernel(
        km_test,
        kp_test,
        kt_test,
        kr_test,
        beta_mechanics=beta[0],
        beta_pressure=beta[1],
        beta_interaction=beta[2],
        rho=rho,
        unit_diagonal=False,
    )
    del km_train, km_test, kp_train, kp_test, kt_train, kt_test, kr_train, kr_test
    gc.collect()
    kernel_elapsed = time.perf_counter() - kernel_started

    weights = {
        "specimen_balanced": specimen_balanced_weights(train["specimen_id"]),
        "rate_balanced": rate_balanced_weights(train["specimen_id"], train["rate_group"]),
    }
    models: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    threshold_scores: dict[str, float] = {}
    train_probability: dict[str, np.ndarray] = {}
    test_probability: dict[str, np.ndarray] = {}
    fit_elapsed: dict[str, float] = {}
    for name, sample_weight in weights.items():
        fit_started = time.perf_counter()
        model = fit_kernel_logistic_regression(
            composite_train,
            train["labels"],
            sample_weight=sample_weight,
            regularization=float(chosen["regularization"]),
            max_iterations=int(config["optimization"]["maximum_iterations"]),
            gradient_tolerance=float(config["optimization"]["gradient_tolerance"]),
        )
        fit_elapsed[name] = time.perf_counter() - fit_started
        train_prob = model.predict_proba(composite_train)
        if selection is None:
            threshold, threshold_score = select_training_threshold(
                train["labels"], train_prob, sample_weight
            )
        else:
            threshold = float(selection["thresholds"][name]["threshold"])
            threshold_score = float(
                selection["thresholds"][name]["weighted_balanced_accuracy"]
            )
        models[name] = model
        thresholds[name] = threshold
        threshold_scores[name] = threshold_score
        train_probability[name] = train_prob
        test_probability[name] = model.predict_proba(composite_test)

    prediction_path = output_dir / "held_out_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "held_out_row_0_based",
            "specimen_id",
            "source_tensor_index_0_based",
            "endpoint_pressure_frame",
            "endpoint_time_s",
            "label",
            "specimen_balanced_probability",
            "specimen_balanced_prediction",
            "rate_balanced_probability",
            "rate_balanced_prediction",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(test["labels"])):
            writer.writerow(
                {
                    "held_out_row_0_based": index,
                    "specimen_id": test_id,
                    "source_tensor_index_0_based": int(test["tensor_indices"][index]),
                    "endpoint_pressure_frame": int(test["endpoint_pressure_frame"][index]),
                    "endpoint_time_s": float(test["endpoint_time_s"][index]),
                    "label": int(test["labels"][index]),
                    "specimen_balanced_probability": float(test_probability["specimen_balanced"][index]),
                    "specimen_balanced_prediction": int(
                        test_probability["specimen_balanced"][index] >= thresholds["specimen_balanced"]
                    ),
                    "rate_balanced_probability": float(test_probability["rate_balanced"][index]),
                    "rate_balanced_prediction": int(
                        test_probability["rate_balanced"][index] >= thresholds["rate_balanced"]
                    ),
                }
            )

    np.savez_compressed(
        output_dir / "fold_model.npz",
        train_specimen_id=train["specimen_id"],
        train_source_tensor_index_0_based=train["tensor_indices"],
        train_labels=train["labels"],
        mechanics_median=mechanics_standardizer.median,
        mechanics_scale=mechanics_standardizer.scale,
        mechanics_feature_names=np.asarray(mechanics_feature_names),
        pressure_channel_rms=pressure_rms,
        phase_median=phase_standardizer.median,
        phase_scale=phase_standardizer.scale,
        phase_feature_name=np.asarray(phase_feature_name),
        specimen_balanced_alpha=models["specimen_balanced"].alpha,
        specimen_balanced_intercept=np.float64(models["specimen_balanced"].intercept),
        specimen_balanced_threshold=np.float64(thresholds["specimen_balanced"]),
        rate_balanced_alpha=models["rate_balanced"].alpha,
        rate_balanced_intercept=np.float64(models["rate_balanced"].intercept),
        rate_balanced_threshold=np.float64(thresholds["rate_balanced"]),
    )
    plot_fold(
        output_dir / "fold_diagnostics.png",
        test_id,
        test["endpoint_time_s"],
        test["labels"],
        test_probability,
        thresholds,
        test_specimen.event_interval_s,
        horizon_ms,
        {name: model.objective_history for name, model in models.items()},
    )

    train_counts = {
        item.specimen_id: {
            "raw_eligible_definite": int(len(item.labels)),
            "selected": int(len(selected_local[item.specimen_id])),
            "selected_positive": int(
                np.count_nonzero(item.labels[selected_local[item.specimen_id]] == 1)
            ),
            "selected_negative": int(
                np.count_nonzero(item.labels[selected_local[item.specimen_id]] == 0)
            ),
            "rate_group": item.rate_group,
        }
        for item in train_specimens
    }
    result = {
        "schema_version": "1.0",
        "status": "pass" if all(model.optimizer_success for model in models.values()) else "optimizer_warning",
        "scope": (
            "single complete LOSO fold with grouped inner selection"
            if selection is not None
            else "single complete LOSO pilot fold with prespecified hyperparameters"
        ),
        "fold_id": args.fold_id,
        "held_out_specimen_id": test_id,
        "train_specimen_ids": fold["train_specimen_ids"],
        "target": {
            "event_id": event_id,
            "prediction_horizon_ms": horizon_ms,
            "interval_label_policy": config["target"]["interval_label_policy"],
            "held_out_event_interval_s": list(test_specimen.event_interval_s),
        },
        "sampling": {
            "policy": config["sampling"]["training_policy"],
            "maximum_training_samples": int(config["sampling"]["maximum_training_samples_per_fold"]),
            "raw_training_eligible_definite": int(sum(len(item.labels) for item in train_specimens)),
            "selected_training_samples": int(len(train["labels"])),
            "cap_triggered": bool(sum(len(item.labels) for item in train_specimens) > len(train["labels"])),
            "per_specimen": train_counts,
            "held_out_policy": config["sampling"]["test_policy"],
            "held_out_samples": int(len(test["labels"])),
        },
        "leakage_controls": {
            "held_out_absent_from_training": bool(test_id not in set(train["specimen_id"].astype(str))),
            "mechanics_standardizer_fit_on_training_only": True,
            "pressure_rms_fit_on_training_only": True,
            "phase_standardizer_fit_on_training_only": True,
            "all_kernel_length_scales_fit_on_training_only": True,
            "sample_weights_fit_on_training_only": True,
            "decision_thresholds_fit_on_training_only": True,
            "held_out_windows_retained_without_sampling": True,
        },
        "kernel": {
            "dtype": "float32",
            "chunk_size": chunk_size,
            "train_shape": list(composite_train.shape),
            "test_train_shape": list(composite_test.shape),
            "hyperparameters": chosen,
            "hyperparameter_source": (
                str(args.selection_json.resolve())
                if selection is not None
                else "fixed prespecified pilot configuration"
            ),
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
            "spatial_origin_kernel": "all ones because no reliable spatial origin label is available",
        },
        "fits": {},
        "timing_seconds": {
            "kernel_construction": float(kernel_elapsed),
            "model_fit": {name: float(value) for name, value in fit_elapsed.items()},
            "total": float(time.perf_counter() - started),
        },
        "outputs": {
            "training_manifest_csv": str(output_dir / "training_sample_manifest.csv"),
            "held_out_predictions_csv": str(prediction_path),
            "model_npz": str(output_dir / "fold_model.npz"),
            "diagnostic_png": str(output_dir / "fold_diagnostics.png"),
        },
    }
    for name, model in models.items():
        result["fits"][name] = {
            "model": model_summary(
                model,
                thresholds[name],
                threshold_scores[name],
                (
                    "grouped inner out-of-fold predictions from outer training specimens"
                    if selection is not None
                    else "fitted probabilities from the outer training fold"
                ),
            ),
            "training_fitted_diagnostic": metric_summary(
                train["labels"], train_probability[name], thresholds[name]
            ),
            "held_out_metrics": metric_summary(
                test["labels"], test_probability[name], thresholds[name]
            ),
            "held_out_event_timing": event_timing_summary(
                test["endpoint_time_s"],
                test_probability[name],
                thresholds[name],
                horizon_ms / 1000.0,
                test_specimen.event_interval_s,
            ),
        }
    (output_dir / "fold_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "fold_id": args.fold_id,
        "held_out_specimen_id": test_id,
        "training_samples": len(train["labels"]),
        "held_out_samples": len(test["labels"]),
        "held_out_metrics": {
            name: result["fits"][name]["held_out_metrics"] for name in models
        },
        "output_dir": str(output_dir),
        "total_seconds": result["timing_seconds"]["total"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
