#!/usr/bin/env python3
"""Run a training-only KLR smoke test on one specimen's processed windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from kernel_logistic_regression import (
    fit_kernel_logistic_regression,
    rate_balanced_weights,
    specimen_balanced_weights,
)


def interval_label(
    endpoint_s: float, event_start_s: float, event_end_s: float, horizon_s: float
) -> int | None:
    horizon_end = endpoint_s + horizon_s
    if horizon_end < event_start_s:
        return 0
    if horizon_end >= event_end_s:
        return 1
    return None


def model_summary(model, probabilities: np.ndarray) -> dict[str, object]:
    return {
        "optimizer_success": model.optimizer_success,
        "optimizer_status": model.optimizer_status,
        "optimizer_message": model.optimizer_message,
        "iterations": model.iterations,
        "function_evaluations": model.function_evaluations,
        "initial_objective": model.initial_objective,
        "final_objective": model.final_objective,
        "objective_reduction": model.initial_objective - model.final_objective,
        "intercept": model.intercept,
        "alpha_l2_norm": float(np.linalg.norm(model.alpha)),
        "probability_minimum": float(np.min(probabilities)),
        "probability_maximum": float(np.max(probabilities)),
    }


def binary_ranking_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = np.asarray(scores)[np.asarray(labels) == 1]
    negative = np.asarray(scores)[np.asarray(labels) == 0]
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Both classes are required for a ranking AUC")
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))


def plot_smoke_test(
    endpoint_times: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    event_interval: tuple[float, float],
    objective_history: np.ndarray,
    specimen_id: str,
    event_id: str,
    horizon_ms: int,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(
        endpoint_times,
        probabilities,
        marker="o",
        color="#1f4e79",
        linewidth=1.7,
        label="Training fitted probability",
    )
    axes[0].scatter(
        endpoint_times,
        labels,
        color=np.where(labels == 1, "#70ad47", "#c00000"),
        marker="s",
        s=38,
        label=f"{horizon_ms} ms target label",
        zorder=3,
    )
    axes[0].axvspan(event_interval[0], event_interval[1], color="#c00000", alpha=0.14)
    axes[0].set_xlabel("Causal-window endpoint time (s)")
    axes[0].set_ylabel("Probability or label")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"{event_id} training smoke test")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        np.arange(len(objective_history)),
        objective_history,
        marker="o",
        markersize=3,
        color="#7030a0",
    )
    axes[1].set_xlabel("L-BFGS callback step")
    axes[1].set_ylabel("Weighted objective")
    axes[1].set_title("Optimization convergence")
    axes[1].grid(alpha=0.2)
    fig.suptitle(f"{specimen_id} kernel logistic regression numerical smoke test", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--kernels-npz", type=Path, required=True)
    parser.add_argument("--event-metadata-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id", default="first_visible_failure_interval")
    parser.add_argument("--horizon-ms", type=int, default=120)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    windows = np.load(args.windows_npz)
    kernels = np.load(args.kernels_npz)
    event_metadata = json.loads(args.event_metadata_json.read_text(encoding="utf-8"))
    specimen_ids_all = np.unique(windows["specimen_id"].astype(str))
    if len(specimen_ids_all) != 1:
        raise ValueError("This smoke test expects windows from exactly one specimen")
    specimen_id = str(specimen_ids_all[0])
    if str(event_metadata["specimen_id"]) != specimen_id:
        raise ValueError("Window tensor and event metadata specimen IDs do not match")
    events = event_metadata["target_definition"]["events"]
    if args.event_id not in events:
        raise KeyError(f"Event is not present in metadata: {args.event_id}")
    event = events[args.event_id]
    event_start, event_end = [float(value) for value in event["target_time_interval_s"]]
    endpoint_times_all = np.asarray(windows["endpoint_time_s"], dtype=np.float64)
    eligible = np.flatnonzero(endpoint_times_all < event_start)
    labels_list = [
        interval_label(
            float(endpoint_times_all[index]),
            event_start,
            event_end,
            args.horizon_ms / 1000.0,
        )
        for index in eligible
    ]
    definite = np.asarray([label is not None for label in labels_list], dtype=bool)
    selected_indices = eligible[definite]
    labels = np.asarray([label for label in labels_list if label is not None], dtype=np.int32)
    if len(np.unique(labels)) < 2:
        raise ValueError("The selected event and horizon do not provide both label classes")

    full_kernel = np.asarray(kernels["composite_kernel"], dtype=np.float32)
    if full_kernel.shape != (len(endpoint_times_all), len(endpoint_times_all)):
        raise ValueError("Composite kernel shape does not match the window tensor")
    kernel = full_kernel[np.ix_(selected_indices, selected_indices)]
    specimen_ids = windows["specimen_id"][selected_indices].astype(str)
    rate_groups = np.asarray(["single-specimen-smoke"] * len(selected_indices))
    specimen_weights = specimen_balanced_weights(specimen_ids)
    rate_weights = rate_balanced_weights(specimen_ids, rate_groups)

    specimen_model = fit_kernel_logistic_regression(
        kernel,
        labels,
        sample_weight=specimen_weights,
        regularization=args.regularization,
        max_iterations=args.max_iterations,
        gradient_tolerance=args.gradient_tolerance,
    )
    rate_model = fit_kernel_logistic_regression(
        kernel,
        labels,
        sample_weight=rate_weights,
        regularization=args.regularization,
        max_iterations=args.max_iterations,
        gradient_tolerance=args.gradient_tolerance,
    )
    specimen_probabilities = specimen_model.predict_proba(kernel)
    rate_probabilities = rate_model.predict_proba(kernel)
    training_auc = binary_ranking_auc(labels, specimen_probabilities)
    positive_minimum = float(np.min(specimen_probabilities[labels == 1]))
    negative_maximum = float(np.max(specimen_probabilities[labels == 0]))

    output_npz = args.output_dir / "klr_smoke_model.npz"
    output_json = args.output_dir / "klr_smoke_test.json"
    output_png = args.output_dir / "klr_smoke_test.png"
    output_csv = args.output_dir / "klr_training_predictions.csv"
    np.savez_compressed(
        output_npz,
        selected_window_indices_0_based=selected_indices.astype(np.int32),
        endpoint_time_s=endpoint_times_all[selected_indices],
        labels=labels,
        specimen_balanced_weights=specimen_weights,
        rate_balanced_weights=rate_weights,
        specimen_balanced_alpha=specimen_model.alpha,
        specimen_balanced_intercept=np.float64(specimen_model.intercept),
        rate_balanced_alpha=rate_model.alpha,
        rate_balanced_intercept=np.float64(rate_model.intercept),
        specimen_balanced_training_probability=specimen_probabilities,
        rate_balanced_training_probability=rate_probabilities,
    )
    plot_smoke_test(
        endpoint_times_all[selected_indices],
        labels,
        specimen_probabilities,
        (event_start, event_end),
        specimen_model.objective_history,
        specimen_id,
        args.event_id,
        args.horizon_ms,
        output_png,
    )

    prediction_records = [
        {
            "window_tensor_index_0_based": int(index),
            "endpoint_time_s": float(endpoint_times_all[index]),
            f"label_event_within_{args.horizon_ms}ms": int(label),
            "specimen_balanced_weight": float(specimen_weights[position]),
            "training_fitted_probability": float(specimen_probabilities[position]),
        }
        for position, (index, label) in enumerate(zip(selected_indices, labels))
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_records[0]))
        writer.writeheader()
        writer.writerows(prediction_records)
    summary = {
        "schema_version": "1.0",
        "specimen_id": specimen_id,
        "status": "training-only numerical smoke test; no held-out specimen and no performance claim",
        "target": {
            "event_id": args.event_id,
            "event_time_interval_s": [event_start, event_end],
            "prediction_horizon_ms": args.horizon_ms,
            "eligible_definite_window_count": int(len(labels)),
            "positive_count": int(np.count_nonzero(labels == 1)),
            "negative_count": int(np.count_nonzero(labels == 0)),
        },
        "kernel": {
            "source": f"{specimen_id} prototype composite kernel",
            "shape": list(kernel.shape),
            "regularization": args.regularization,
            "optimizer": "L-BFGS-B",
            "maximum_iterations": args.max_iterations,
            "gradient_tolerance": args.gradient_tolerance,
            "full_hessian_formed": False,
        },
        "specimen_balanced_fit": model_summary(specimen_model, specimen_probabilities),
        "rate_balanced_fit": model_summary(rate_model, rate_probabilities),
        "weight_comparison": {
            "maximum_absolute_difference": float(
                np.max(np.abs(specimen_weights - rate_weights))
            ),
            "interpretation": "identical for a single specimen in a single provisional rate group",
        },
        "training_ranking_diagnostic": {
            "auc": training_auc,
            "positive_probability_minimum": positive_minimum,
            "negative_probability_maximum": negative_maximum,
            "positive_negative_separation_margin": positive_minimum - negative_maximum,
            "fixed_0_5_threshold_positive_count": int(
                np.count_nonzero(specimen_probabilities >= 0.5)
            ),
            "interpretation": (
                "training-only ranking diagnostic, not a generalization metric; the decision "
                "threshold must be selected inside each future training fold rather than fixed at 0.5"
            ),
        },
        "predictions": prediction_records,
        "checks": {
            "both_optimizers_converged": bool(
                specimen_model.optimizer_success and rate_model.optimizer_success
            ),
            "both_objectives_decreased": bool(
                specimen_model.final_objective < specimen_model.initial_objective
                and rate_model.final_objective < rate_model.initial_objective
            ),
            "probabilities_finite_and_bounded": bool(
                np.all(np.isfinite(specimen_probabilities))
                and np.all((specimen_probabilities >= 0) & (specimen_probabilities <= 1))
            ),
        },
        "outputs": {
            "model_npz": str(output_npz.resolve()),
            "diagnostic_figure": str(output_png.resolve()),
            "training_predictions_csv": str(output_csv.resolve()),
        },
    }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    compact_summary = {key: value for key, value in summary.items() if key != "predictions"}
    compact_summary["prediction_record_count"] = len(prediction_records)
    print(json.dumps(compact_summary, ensure_ascii=False, indent=2))
    if not all(summary["checks"].values()):
        raise RuntimeError("KLR smoke test failed one or more numerical checks")


if __name__ == "__main__":
    main()
