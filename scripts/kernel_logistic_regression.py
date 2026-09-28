#!/usr/bin/env python3
"""Weighted kernel logistic regression using a memory-safe L-BFGS optimizer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import project_dependencies  # noqa: F401
from scipy.optimize import minimize


@dataclass(frozen=True)
class KernelLogisticModel:
    alpha: np.ndarray
    intercept: float
    regularization: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    iterations: int
    function_evaluations: int
    initial_objective: float
    final_objective: float
    objective_history: np.ndarray

    def decision_function(self, kernel_test_train: np.ndarray) -> np.ndarray:
        kernel = np.asarray(kernel_test_train, dtype=np.float32)
        if kernel.ndim != 2 or kernel.shape[1] != len(self.alpha):
            raise ValueError("kernel_test_train must have one column per training sample")
        return kernel @ self.alpha + self.intercept

    def predict_proba(self, kernel_test_train: np.ndarray) -> np.ndarray:
        return stable_sigmoid(self.decision_function(kernel_test_train))


def stable_sigmoid(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_scores = np.exp(values[~positive])
    output[~positive] = exp_scores / (1.0 + exp_scores)
    return output


def normalize_sample_weights(sample_weight: np.ndarray | None, size: int) -> np.ndarray:
    if sample_weight is None:
        return np.full(size, 1.0 / size, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    if weights.shape != (size,):
        raise ValueError("sample_weight length must equal the number of samples")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("sample weights must have a positive sum")
    return weights / total


def specimen_balanced_weights(specimen_ids: np.ndarray) -> np.ndarray:
    specimen = np.asarray(specimen_ids).astype(str).reshape(-1)
    if len(specimen) == 0:
        raise ValueError("at least one sample is required")
    unique, counts = np.unique(specimen, return_counts=True)
    count_by_specimen = dict(zip(unique.tolist(), counts.tolist()))
    raw = np.asarray([1.0 / count_by_specimen[value] for value in specimen], dtype=np.float64)
    return raw / np.sum(raw)


def rate_balanced_weights(specimen_ids: np.ndarray, rate_groups: np.ndarray) -> np.ndarray:
    specimen = np.asarray(specimen_ids).astype(str).reshape(-1)
    groups = np.asarray(rate_groups).astype(str).reshape(-1)
    if specimen.shape != groups.shape or len(specimen) == 0:
        raise ValueError("specimen IDs and rate groups must have matching non-empty shapes")

    unique_specimens, specimen_counts = np.unique(specimen, return_counts=True)
    specimen_sample_count = dict(zip(unique_specimens.tolist(), specimen_counts.tolist()))
    specimen_group: dict[str, str] = {}
    for specimen_id in unique_specimens:
        assigned = np.unique(groups[specimen == specimen_id])
        if len(assigned) != 1:
            raise ValueError(f"Specimen {specimen_id} belongs to multiple rate groups")
        specimen_group[str(specimen_id)] = str(assigned[0])
    group_specimen_count: dict[str, int] = {}
    for group in specimen_group.values():
        group_specimen_count[group] = group_specimen_count.get(group, 0) + 1
    raw = np.asarray(
        [
            1.0
            / (
                group_specimen_count[specimen_group[specimen_id]]
                * specimen_sample_count[specimen_id]
            )
            for specimen_id in specimen
        ],
        dtype=np.float64,
    )
    return raw / np.sum(raw)


def _validate_training_inputs(
    kernel_train: np.ndarray, labels: np.ndarray, sample_weight: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernel = np.asarray(kernel_train, dtype=np.float32)
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel_train must be a square matrix")
    if not np.all(np.isfinite(kernel)):
        raise ValueError("kernel_train must be finite")
    symmetry_error = float(np.max(np.abs(kernel - kernel.T)))
    if symmetry_error > 1.0e-5:
        raise ValueError(f"kernel_train must be symmetric; max error={symmetry_error}")
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if target.shape != (len(kernel),):
        raise ValueError("labels length must match kernel_train")
    if not np.all(np.isin(target, [0.0, 1.0])):
        raise ValueError("labels must contain only 0 and 1")
    if len(np.unique(target)) < 2:
        raise ValueError("kernel logistic regression requires both label classes")
    weights = normalize_sample_weights(sample_weight, len(kernel))
    return kernel, target, weights


def objective_and_gradient(
    parameters: np.ndarray,
    kernel: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    sample_count = len(labels)
    alpha = parameters[:sample_count]
    intercept = float(parameters[-1])
    kernel_alpha = kernel @ alpha
    scores = kernel_alpha + intercept
    data_loss = float(np.sum(weights * (np.logaddexp(0.0, scores) - labels * scores)))
    regularization_loss = 0.5 * regularization * float(alpha @ kernel_alpha)
    residual = weights * (stable_sigmoid(scores) - labels)
    gradient_alpha = kernel.T @ residual + regularization * kernel_alpha
    gradient_intercept = float(np.sum(residual))
    gradient = np.concatenate([gradient_alpha, np.asarray([gradient_intercept])])
    return data_loss + regularization_loss, gradient


def fit_kernel_logistic_regression(
    kernel_train: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    regularization: float = 0.1,
    max_iterations: int = 500,
    gradient_tolerance: float = 1.0e-7,
) -> KernelLogisticModel:
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    kernel, target, weights = _validate_training_inputs(
        kernel_train, labels, sample_weight
    )
    positive_prior = float(np.clip(np.sum(weights * target), 1.0e-6, 1.0 - 1.0e-6))
    initial_intercept = float(np.log(positive_prior / (1.0 - positive_prior)))
    initial = np.zeros(len(kernel) + 1, dtype=np.float64)
    initial[-1] = initial_intercept
    initial_objective, _ = objective_and_gradient(
        initial, kernel, target, weights, regularization
    )
    history = [initial_objective]

    def callback(parameters: np.ndarray) -> None:
        value, _ = objective_and_gradient(
            parameters, kernel, target, weights, regularization
        )
        history.append(float(value))

    result = minimize(
        objective_and_gradient,
        initial,
        args=(kernel, target, weights, regularization),
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={
            "maxiter": int(max_iterations),
            "gtol": float(gradient_tolerance),
            "ftol": 1.0e-12,
            "maxls": 40,
        },
    )
    final_objective, _ = objective_and_gradient(
        result.x, kernel, target, weights, regularization
    )
    if not history or not np.isclose(history[-1], final_objective):
        history.append(float(final_objective))
    return KernelLogisticModel(
        alpha=np.asarray(result.x[:-1], dtype=np.float64),
        intercept=float(result.x[-1]),
        regularization=float(regularization),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        function_evaluations=int(result.nfev),
        initial_objective=float(initial_objective),
        final_objective=float(final_objective),
        objective_history=np.asarray(history, dtype=np.float64),
    )
