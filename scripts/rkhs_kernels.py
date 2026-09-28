#!/usr/bin/env python3
"""Memory-aware CPU reference kernels for the ice-load RKHS model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DTYPE = np.float32
DEFAULT_CHUNK_SIZE = 256


@dataclass(frozen=True)
class RobustStandardizer:
    median: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class KernelResult:
    kernel: np.ndarray
    length_scale: float
    squared_distance: np.ndarray
    degenerate: bool


@dataclass(frozen=True)
class PressureKernelResult:
    kernel: np.ndarray
    pressure_length_scale: float
    gradient_length_scale: float
    pressure_squared_distance: np.ndarray
    gradient_squared_distance: np.ndarray
    pressure_channel_rms: np.ndarray
    degenerate_pressure: bool
    degenerate_gradient: bool


def _as_float32_2d(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=DTYPE)
    if result.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Kernel vectors must be finite")
    return result


def squared_euclidean_blockwise(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Compute pairwise squared Euclidean distance without an N1 x N2 x d tensor."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    x_array = _as_float32_2d(x)
    symmetric = y is None
    y_array = x_array if symmetric else _as_float32_2d(y)
    if x_array.shape[1] != y_array.shape[1]:
        raise ValueError("x and y must have the same feature dimension")

    x_norm = np.einsum("ij,ij->i", x_array, x_array, dtype=DTYPE)
    y_norm = x_norm if symmetric else np.einsum(
        "ij,ij->i", y_array, y_array, dtype=DTYPE
    )
    output = np.empty((len(x_array), len(y_array)), dtype=DTYPE)

    if symmetric:
        for i0 in range(0, len(x_array), chunk_size):
            i1 = min(i0 + chunk_size, len(x_array))
            xi = x_array[i0:i1]
            for j0 in range(i0, len(y_array), chunk_size):
                j1 = min(j0 + chunk_size, len(y_array))
                block = x_norm[i0:i1, None] + y_norm[None, j0:j1]
                block -= DTYPE(2.0) * (xi @ y_array[j0:j1].T)
                np.maximum(block, DTYPE(0.0), out=block)
                output[i0:i1, j0:j1] = block
                if j0 != i0:
                    output[j0:j1, i0:i1] = block.T
        np.fill_diagonal(output, DTYPE(0.0))
    else:
        for i0 in range(0, len(x_array), chunk_size):
            i1 = min(i0 + chunk_size, len(x_array))
            block = x_norm[i0:i1, None] + y_norm[None, :]
            block -= DTYPE(2.0) * (x_array[i0:i1] @ y_array.T)
            np.maximum(block, DTYPE(0.0), out=block)
            output[i0:i1] = block
    return output


def median_length_scale_from_squared_distance(
    squared_distance: np.ndarray,
    *,
    seed: int = 20260915,
    max_pairs: int = 200_000,
) -> tuple[float, bool]:
    """Median non-zero Euclidean distance with bounded-memory pair sampling."""
    distance = np.asarray(squared_distance)
    if distance.ndim != 2:
        raise ValueError("squared_distance must be a matrix")
    if distance.shape[0] == distance.shape[1]:
        size = distance.shape[0]
        count = size * (size - 1) // 2
        if count > max_pairs:
            rng = np.random.default_rng(seed)
            row = rng.integers(0, size, size=max_pairs, dtype=np.int32)
            col = rng.integers(0, size - 1, size=max_pairs, dtype=np.int32)
            col += col >= row
            values = distance[row, col]
        else:
            row, col = np.triu_indices(size, k=1)
            values = distance[row, col]
    else:
        flat = distance.ravel()
        if flat.size > max_pairs:
            rng = np.random.default_rng(seed)
            values = flat[rng.choice(flat.size, size=max_pairs, replace=False)]
        else:
            values = flat
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0, True
    scale = float(np.sqrt(np.median(positive.astype(np.float64))))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0, True
    return scale, False


def rbf_from_squared_distance(
    squared_distance: np.ndarray,
    length_scale: float,
    *,
    copy: bool = True,
    unit_diagonal: bool | None = None,
) -> np.ndarray:
    if not np.isfinite(length_scale) or length_scale <= 0:
        raise ValueError("length_scale must be finite and positive")
    kernel = np.array(squared_distance, dtype=DTYPE, copy=copy)
    kernel *= DTYPE(-0.5 / (length_scale * length_scale))
    np.exp(kernel, out=kernel)
    if unit_diagonal is None:
        unit_diagonal = kernel.shape[0] == kernel.shape[1]
    if unit_diagonal:
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("unit_diagonal requires a square matrix")
        np.fill_diagonal(kernel, DTYPE(1.0))
    return kernel


def rbf_kernel(
    vectors: np.ndarray,
    *,
    length_scale: float | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> KernelResult:
    distance = squared_euclidean_blockwise(vectors, chunk_size=chunk_size)
    if length_scale is None:
        length_scale, degenerate = median_length_scale_from_squared_distance(distance)
    else:
        degenerate = False
    if degenerate:
        kernel = np.ones_like(distance, dtype=DTYPE)
    else:
        kernel = rbf_from_squared_distance(distance, length_scale)
    return KernelResult(kernel, float(length_scale), distance, degenerate)


def fit_robust_standardizer(
    values: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    training_indices: np.ndarray | list[int] | None = None,
) -> RobustStandardizer:
    """Fit per-feature median/IQR using training windows only."""
    array = np.asarray(values, dtype=DTYPE)
    if array.ndim < 2:
        raise ValueError("values must include a feature axis")
    if valid_mask is None:
        valid = np.isfinite(array)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(array)
        if valid.shape != array.shape:
            raise ValueError("valid_mask shape must match values")
    if training_indices is not None:
        array = array[np.asarray(training_indices, dtype=int)]
        valid = valid[np.asarray(training_indices, dtype=int)]

    flat_values = array.reshape(-1, array.shape[-1])
    flat_valid = valid.reshape(-1, valid.shape[-1])
    medians = np.empty(array.shape[-1], dtype=DTYPE)
    scales = np.empty(array.shape[-1], dtype=DTYPE)
    for feature in range(array.shape[-1]):
        selected = flat_values[flat_valid[:, feature], feature]
        if selected.size == 0:
            raise ValueError(f"Feature {feature} has no valid training value")
        median = float(np.median(selected.astype(np.float64)))
        q25, q75 = np.quantile(selected.astype(np.float64), [0.25, 0.75])
        scale = float(q75 - q25)
        if not np.isfinite(scale) or scale <= 1.0e-8:
            scale = float(np.std(selected.astype(np.float64)))
        if not np.isfinite(scale) or scale <= 1.0e-8:
            scale = 1.0
        medians[feature] = DTYPE(median)
        scales[feature] = DTYPE(scale)
    return RobustStandardizer(medians, scales)


def transform_and_flatten_sequence_features(
    values: np.ndarray,
    valid_mask: np.ndarray,
    standardizer: RobustStandardizer,
) -> np.ndarray:
    array = np.asarray(values, dtype=DTYPE)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(array)
    if valid.shape != array.shape:
        raise ValueError("valid_mask shape must match values")
    filled = np.where(valid, array, standardizer.median).astype(DTYPE, copy=False)
    filled -= standardizer.median
    filled /= standardizer.scale
    flattened = filled.reshape(len(filled), -1)
    flattened /= DTYPE(np.sqrt(flattened.shape[1]))
    return flattened


def mechanics_kernel(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    training_indices: np.ndarray | list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[KernelResult, RobustStandardizer, np.ndarray]:
    standardizer = fit_robust_standardizer(
        values, valid_mask, training_indices=training_indices
    )
    vectors = transform_and_flatten_sequence_features(values, valid_mask, standardizer)
    training_vectors = vectors if training_indices is None else vectors[np.asarray(training_indices)]
    training_distance = squared_euclidean_blockwise(training_vectors, chunk_size=chunk_size)
    length_scale, degenerate = median_length_scale_from_squared_distance(training_distance)
    full_distance = squared_euclidean_blockwise(vectors, chunk_size=chunk_size)
    kernel = (
        np.ones_like(full_distance, dtype=DTYPE)
        if degenerate
        else rbf_from_squared_distance(full_distance, length_scale)
    )
    return KernelResult(kernel, length_scale, full_distance, degenerate), standardizer, vectors


def fit_pressure_channel_rms(
    pressure_channels: np.ndarray,
    footprint_overlap_fraction: np.ndarray,
    *,
    training_indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    """Fit one weighted RMS scale for pressure and one for both gradients."""
    fields = np.asarray(pressure_channels, dtype=DTYPE)
    if fields.ndim != 5 or fields.shape[2] != 3:
        raise ValueError("pressure_channels must have shape (N, H, 3, Y, X)")
    if training_indices is not None:
        fields = fields[np.asarray(training_indices, dtype=int)]
    weights = np.asarray(footprint_overlap_fraction, dtype=DTYPE)
    if weights.shape != fields.shape[-2:]:
        raise ValueError("footprint_overlap_fraction shape must match the field grid")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        raise ValueError("footprint overlap weights must sum to a positive value")
    denominator_pressure = fields.shape[0] * fields.shape[1] * weight_sum
    denominator_gradient = denominator_pressure * 2.0
    pressure_sum_squares = 0.0
    gradient_sum_squares = 0.0
    for start in range(0, len(fields), 64):
        block = fields[start : start + 64]
        pressure_sum_squares += float(
            np.einsum(
                "nhyx,yx,nhyx->",
                block[:, :, 0],
                weights,
                block[:, :, 0],
                dtype=np.float64,
                optimize=True,
            )
        )
        gradient_sum_squares += float(
            np.einsum(
                "nhcyx,yx,nhcyx->",
                block[:, :, 1:],
                weights,
                block[:, :, 1:],
                dtype=np.float64,
                optimize=True,
            )
        )
    pressure_rms = float(np.sqrt(pressure_sum_squares / denominator_pressure))
    gradient_rms = float(np.sqrt(gradient_sum_squares / denominator_gradient))
    if not np.isfinite(pressure_rms) or pressure_rms <= 1.0e-8:
        pressure_rms = 1.0
    if not np.isfinite(gradient_rms) or gradient_rms <= 1.0e-8:
        gradient_rms = 1.0
    return np.asarray([pressure_rms, gradient_rms], dtype=DTYPE)


def prepare_pressure_vectors(
    pressure_channels: np.ndarray,
    footprint_overlap_fraction: np.ndarray,
    channel_rms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fields = np.asarray(pressure_channels, dtype=DTYPE)
    weights = np.asarray(footprint_overlap_fraction, dtype=DTYPE)
    rms = np.asarray(channel_rms, dtype=DTYPE)
    if fields.ndim != 5 or fields.shape[2] != 3:
        raise ValueError("pressure_channels must have shape (N, H, 3, Y, X)")
    if rms.shape != (2,):
        raise ValueError("channel_rms must contain pressure and gradient scales")
    history = fields.shape[1]
    weighted_pressure = DTYPE(1.0) / DTYPE(
        np.sqrt(history * float(np.sum(weights)))
    )
    weighted_gradient = DTYPE(1.0) / DTYPE(
        np.sqrt(2.0 * history * float(np.sum(weights)))
    )
    sqrt_weight = np.sqrt(weights).astype(DTYPE, copy=False)
    pressure_vector = (
        fields[:, :, 0] * sqrt_weight[None, None, :, :] / rms[0] * weighted_pressure
    ).reshape(len(fields), -1)
    gradient_vector = (
        fields[:, :, 1:] * sqrt_weight[None, None, None, :, :] / rms[1] * weighted_gradient
    ).reshape(len(fields), -1)
    return pressure_vector.astype(DTYPE, copy=False), gradient_vector.astype(DTYPE, copy=False)


def pressure_field_kernel(
    pressure_channels: np.ndarray,
    footprint_overlap_fraction: np.ndarray,
    *,
    training_indices: np.ndarray | list[int] | None = None,
    pressure_weight: float = 0.5,
    gradient_weight: float = 0.5,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> PressureKernelResult:
    if pressure_weight < 0 or gradient_weight < 0:
        raise ValueError("pressure and gradient weights must be nonnegative")
    if pressure_weight + gradient_weight <= 0:
        raise ValueError("at least one pressure-kernel component must be positive")
    total_weight = pressure_weight + gradient_weight
    pressure_weight /= total_weight
    gradient_weight /= total_weight
    channel_rms = fit_pressure_channel_rms(
        pressure_channels,
        footprint_overlap_fraction,
        training_indices=training_indices,
    )
    pressure_vector, gradient_vector = prepare_pressure_vectors(
        pressure_channels, footprint_overlap_fraction, channel_rms
    )
    train_pressure = (
        pressure_vector
        if training_indices is None
        else pressure_vector[np.asarray(training_indices, dtype=int)]
    )
    train_gradient = (
        gradient_vector
        if training_indices is None
        else gradient_vector[np.asarray(training_indices, dtype=int)]
    )
    train_d0 = squared_euclidean_blockwise(train_pressure, chunk_size=chunk_size)
    train_d1 = squared_euclidean_blockwise(train_gradient, chunk_size=chunk_size)
    length0, degenerate0 = median_length_scale_from_squared_distance(train_d0)
    length1, degenerate1 = median_length_scale_from_squared_distance(train_d1)
    d0 = squared_euclidean_blockwise(pressure_vector, chunk_size=chunk_size)
    d1 = squared_euclidean_blockwise(gradient_vector, chunk_size=chunk_size)
    exponent = np.zeros_like(d0, dtype=DTYPE)
    if not degenerate0 and pressure_weight > 0:
        exponent += DTYPE(pressure_weight / (length0 * length0)) * d0
    if not degenerate1 and gradient_weight > 0:
        exponent += DTYPE(gradient_weight / (length1 * length1)) * d1
    exponent *= DTYPE(-0.5)
    np.exp(exponent, out=exponent)
    np.fill_diagonal(exponent, DTYPE(1.0))
    return PressureKernelResult(
        exponent,
        length0,
        length1,
        d0,
        d1,
        channel_rms,
        degenerate0,
        degenerate1,
    )


def endpoint_scalar_kernel(
    values: np.ndarray,
    *,
    training_indices: np.ndarray | list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[KernelResult, RobustStandardizer, np.ndarray]:
    scalar = np.asarray(values, dtype=DTYPE).reshape(-1, 1)
    valid = np.isfinite(scalar)
    standardizer = fit_robust_standardizer(
        scalar[:, None, :], valid[:, None, :], training_indices=training_indices
    )
    vectors = transform_and_flatten_sequence_features(
        scalar[:, None, :], valid[:, None, :], standardizer
    )
    training_vectors = vectors if training_indices is None else vectors[np.asarray(training_indices)]
    train_distance = squared_euclidean_blockwise(training_vectors, chunk_size=chunk_size)
    length_scale, degenerate = median_length_scale_from_squared_distance(train_distance)
    full_distance = squared_euclidean_blockwise(vectors, chunk_size=chunk_size)
    kernel = (
        np.ones_like(full_distance, dtype=DTYPE)
        if degenerate
        else rbf_from_squared_distance(full_distance, length_scale)
    )
    return KernelResult(kernel, length_scale, full_distance, degenerate), standardizer, vectors


def loading_rate_kernel(
    log_rate_per_window: np.ndarray,
    specimen_id_per_window: np.ndarray,
    *,
    training_indices: np.ndarray | list[int] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> KernelResult:
    """RBF loading-rate kernel with its scale fitted on unique training specimens."""
    rates = np.asarray(log_rate_per_window, dtype=DTYPE).reshape(-1)
    specimen_ids = np.asarray(specimen_id_per_window).astype(str).reshape(-1)
    if rates.shape != specimen_ids.shape:
        raise ValueError("loading rates and specimen IDs must have matching lengths")
    if not np.all(np.isfinite(rates)):
        raise ValueError("loading rates must be finite")
    train = (
        np.arange(len(rates), dtype=int)
        if training_indices is None
        else np.asarray(training_indices, dtype=int)
    )
    unique_rates = []
    for specimen_id in np.unique(specimen_ids[train]):
        values = rates[train][specimen_ids[train] == specimen_id]
        if not np.allclose(values, values[0], rtol=1.0e-6, atol=1.0e-8):
            raise ValueError(f"Specimen {specimen_id} has inconsistent representative rates")
        unique_rates.append(float(values[0]))
    unique_vectors = np.asarray(unique_rates, dtype=DTYPE).reshape(-1, 1)
    training_distance = squared_euclidean_blockwise(
        unique_vectors, chunk_size=chunk_size
    )
    length_scale, degenerate = median_length_scale_from_squared_distance(training_distance)
    full_distance = squared_euclidean_blockwise(rates.reshape(-1, 1), chunk_size=chunk_size)
    kernel = (
        np.ones_like(full_distance, dtype=DTYPE)
        if degenerate
        else rbf_from_squared_distance(full_distance, length_scale)
    )
    return KernelResult(kernel, length_scale, full_distance, degenerate)


def soft_rate_kernel(rate_kernel: np.ndarray, rho: float) -> np.ndarray:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return (DTYPE(1.0 - rho) + DTYPE(rho) * np.asarray(rate_kernel, dtype=DTYPE)).astype(
        DTYPE, copy=False
    )


def composite_kernel(
    mechanics: np.ndarray,
    pressure: np.ndarray,
    phase: np.ndarray,
    rate: np.ndarray,
    *,
    beta_mechanics: float,
    beta_pressure: float,
    beta_interaction: float,
    rho: float,
    spatial: np.ndarray | None = None,
    unit_diagonal: bool | None = None,
) -> np.ndarray:
    betas = np.asarray(
        [beta_mechanics, beta_pressure, beta_interaction], dtype=np.float64
    )
    if np.any(betas < 0) or not np.isclose(np.sum(betas), 1.0, atol=1.0e-8):
        raise ValueError("nonnegative beta weights must sum to 1")
    km = np.asarray(mechanics, dtype=DTYPE)
    kp = np.asarray(pressure, dtype=DTYPE)
    kt = np.asarray(phase, dtype=DTYPE)
    kr = np.asarray(rate, dtype=DTYPE)
    if not (km.shape == kp.shape == kt.shape == kr.shape):
        raise ValueError("all base kernels must share the same shape")
    ks = np.ones_like(km, dtype=DTYPE) if spatial is None else np.asarray(spatial, dtype=DTYPE)
    if ks.shape != km.shape:
        raise ValueError("spatial kernel must share the base-kernel shape")
    mixture = DTYPE(beta_mechanics) * km
    mixture += DTYPE(beta_pressure) * kp
    mixture += DTYPE(beta_interaction) * (km * kp)
    result = ks * kt
    result *= soft_rate_kernel(kr, rho)
    result *= mixture
    if unit_diagonal is None:
        unit_diagonal = result.shape[0] == result.shape[1]
    if unit_diagonal:
        if result.shape[0] != result.shape[1]:
            raise ValueError("unit_diagonal requires a square matrix")
        np.fill_diagonal(result, DTYPE(1.0))
    return result.astype(DTYPE, copy=False)


def kernel_diagnostics(
    kernel: np.ndarray,
) -> dict[str, float | int | bool | list[int] | None]:
    matrix = np.asarray(kernel, dtype=np.float64)
    symmetric_error = float(np.max(np.abs(matrix - matrix.T)))
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    tolerance = max(1.0e-7, 1.0e-6 * maximum_eigenvalue)
    positive = np.clip(eigenvalues, 0.0, None)
    positive_sum = float(np.sum(positive))
    if positive_sum > 0:
        probabilities = positive / positive_sum
        nonzero_probabilities = probabilities[probabilities > 0]
        effective_rank = float(
            np.exp(-np.sum(nonzero_probabilities * np.log(nonzero_probabilities)))
        )
    else:
        effective_rank = 0.0
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    retained = eigenvalues[eigenvalues > tolerance]
    positive_spectrum_condition = (
        float(retained[-1] / retained[0]) if retained.size else None
    )
    row, col = np.triu_indices(matrix.shape[0], k=1)
    return {
        "shape": list(matrix.shape),
        "symmetric": bool(symmetric_error <= 1.0e-6),
        "maximum_symmetry_error": symmetric_error,
        "diagonal_maximum_error_from_one": float(
            np.max(np.abs(np.diag(matrix) - 1.0))
        ),
        "minimum_value": float(np.min(matrix)),
        "maximum_value": float(np.max(matrix)),
        "minimum_eigenvalue": minimum_eigenvalue,
        "maximum_eigenvalue": maximum_eigenvalue,
        "psd_with_numerical_tolerance": bool(minimum_eigenvalue >= -tolerance),
        "psd_tolerance": tolerance,
        "numerical_rank_at_tolerance": numerical_rank,
        "effective_rank": effective_rank,
        "singular_at_tolerance": bool(numerical_rank < matrix.shape[0]),
        "positive_spectrum_condition_number": positive_spectrum_condition,
        "mean_off_diagonal_similarity": float(np.mean(matrix[row, col]))
        if row.size
        else 1.0,
    }
