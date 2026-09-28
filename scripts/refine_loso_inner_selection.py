#!/usr/bin/env python3
"""Refit the selected inner-LOSO model to a stricter convergence criterion."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import project_dependencies  # noqa: F401

from kernel_logistic_regression import (
    fit_kernel_logistic_regression,
    rate_balanced_weights,
    specimen_balanced_weights,
)
from run_loso_fold import metric_summary, select_training_threshold
from tune_loso_fold import aggregate_score, compose, load_cache


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/loso/loso_v1.json"))
    parser.add_argument("--fold-id", default="fold_01")
    parser.add_argument("--selection-root", type=Path)
    parser.add_argument("--maximum-iterations", type=int, default=4000)
    parser.add_argument("--gradient-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_json(args.config)
    workspace = args.config.resolve().parents[2]
    selection_root = args.selection_root or (
        workspace / config["output_root"] / "fold_runs" / args.fold_id / "inner_selection"
    )
    source = load_json(selection_root / "inner_selection.json")
    selected = source["selected_hyperparameters"]
    beta = [float(value) for value in selected["beta"]]
    rho = float(selected["rho"])
    regularization = float(selected["regularization"])
    split_count = len(source["inner_splits"])
    thresholds = {}
    all_details = {}
    for weighting in ("specimen_balanced", "rate_balanced"):
        accumulated = {
            "labels": [],
            "probabilities": [],
            "specimen_ids": [],
            "rate_groups": [],
        }
        details = []
        for split_number in range(1, split_count + 1):
            kernels, arrays = load_cache(
                selection_root / "kernel_cache" / f"inner_{split_number:02d}"
            )
            kernel_train, kernel_validation = compose(kernels, beta, rho)
            if weighting == "specimen_balanced":
                sample_weight = specimen_balanced_weights(arrays["train_specimen_id"])
            else:
                sample_weight = rate_balanced_weights(
                    arrays["train_specimen_id"], arrays["train_rate_group"]
                )
            model = fit_kernel_logistic_regression(
                kernel_train,
                arrays["train_labels"],
                sample_weight=sample_weight,
                regularization=regularization,
                max_iterations=args.maximum_iterations,
                gradient_tolerance=args.gradient_tolerance,
            )
            accumulated["labels"].append(arrays["validation_labels"])
            accumulated["probabilities"].append(model.predict_proba(kernel_validation))
            accumulated["specimen_ids"].append(arrays["validation_specimen_id"])
            accumulated["rate_groups"].append(arrays["validation_rate_group"])
            details.append(
                {
                    "inner_fold": split_number,
                    "optimizer_success": bool(model.optimizer_success),
                    "optimizer_status": int(model.optimizer_status),
                    "optimizer_message": str(model.optimizer_message),
                    "iterations": int(model.iterations),
                    "function_evaluations": int(model.function_evaluations),
                    "initial_objective": float(model.initial_objective),
                    "final_objective": float(model.final_objective),
                }
            )
            print(
                f"{weighting} inner {split_number}: success={model.optimizer_success}, "
                f"iterations={model.iterations}",
                flush=True,
            )
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
        thresholds[weighting] = {
            "threshold": threshold,
            "weighted_balanced_accuracy": threshold_score,
            "weighted_log_loss": score["weighted_log_loss"],
            "window_auc": score["window_auc"],
            "all_inner_optimizers_converged": all(
                detail["optimizer_success"] for detail in details
            ),
            "unweighted_oof_metrics": metric_summary(
                score["labels"], score["probabilities"], threshold
            ),
        }
        all_details[weighting] = details
    result = dict(source)
    result["status"] = (
        "pass"
        if all(value["all_inner_optimizers_converged"] for value in thresholds.values())
        else "optimizer_warning"
    )
    result["thresholds"] = thresholds
    result["refinement"] = {
        "maximum_iterations": args.maximum_iterations,
        "gradient_tolerance": args.gradient_tolerance,
        "optimizer_details": all_details,
        "total_seconds": float(time.perf_counter() - started),
    }
    output = selection_root / "inner_selection_refined.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_hyperparameters": selected,
                "thresholds": thresholds,
                "output": str(output),
                "total_seconds": result["refinement"]["total_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
