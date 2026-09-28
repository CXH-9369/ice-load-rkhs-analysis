#!/usr/bin/env python3
"""Fit and diagnose the CPU reference kernels on one processed specimen."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from rkhs_kernels import (
    DTYPE,
    composite_kernel,
    endpoint_scalar_kernel,
    kernel_diagnostics,
    loading_rate_kernel,
    mechanics_kernel,
    pressure_field_kernel,
)


def off_diagonal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    row, col = np.triu_indices(left.shape[0], k=1)
    x = np.asarray(left[row, col], dtype=np.float64)
    y = np.asarray(right[row, col], dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def plot_kernels(
    kernels: dict[str, np.ndarray],
    times: np.ndarray,
    specimen_label: str,
    output: Path,
) -> None:
    names = ["K_M", "K_P", "K_T", "K_R", "K_M x K_P", "K_PI prototype"]
    fig = plt.figure(figsize=(15.2, 9.2))
    grid = fig.add_gridspec(
        2, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.34, hspace=0.34
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, column]) for column in range(3)],
            [fig.add_subplot(grid[1, column]) for column in range(3)],
        ]
    )
    colorbar_axis = fig.add_subplot(grid[:, 3])
    image = None
    tick_indices = np.unique(np.linspace(0, len(times) - 1, min(6, len(times))).astype(int))
    tick_labels = [f"{times[index]:.3f}" for index in tick_indices]
    for axis, name in zip(axes.ravel(), names):
        image = axis.imshow(kernels[name], cmap="viridis", vmin=0, vmax=1, origin="upper")
        axis.set_title(name)
        axis.set_xticks(tick_indices, tick_labels, rotation=45, ha="right", fontsize=8)
        axis.set_yticks(tick_indices, tick_labels, fontsize=8)
        axis.set_xlabel("Window endpoint time (s)")
        axis.set_ylabel("Window endpoint time (s)")
    fig.colorbar(image, cax=colorbar_axis, label="Kernel similarity")
    fig.suptitle(f"{specimen_label} CPU reference RKHS kernel matrices", fontsize=15)
    fig.subplots_adjust(left=0.07, right=0.94, bottom=0.09, top=0.92)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--event-metadata-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--beta-mechanics", type=float, default=1.0 / 3.0)
    parser.add_argument("--beta-pressure", type=float, default=1.0 / 3.0)
    parser.add_argument("--beta-interaction", type=float, default=1.0 / 3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(args.windows_npz)
    event_metadata = json.loads(args.event_metadata_json.read_text(encoding="utf-8"))
    specimen_ids = np.unique(source["specimen_id"].astype(str))
    if len(specimen_ids) != 1:
        raise ValueError("This isolated diagnostic expects windows from exactly one specimen")
    specimen_id = str(specimen_ids[0])
    if str(event_metadata["specimen_id"]) != specimen_id:
        raise ValueError("Window tensor and event metadata specimen IDs do not match")
    spatial_origin = event_metadata["target_definition"].get("spatial_origin_label", {})
    if bool(spatial_origin.get("available", False)):
        raise ValueError(
            "A spatial-origin label is available, but this diagnostic has no K_S implementation"
        )
    training_indices = np.arange(len(source["endpoint_time_s"]), dtype=np.int32)
    feature_names = source["mechanics_feature_names"].astype(str).tolist()
    strain_index = feature_names.index(
        "machine_engineering_strain_actual_from_first_record_pct"
    )

    timings: dict[str, float] = {}
    start = time.perf_counter()
    mechanics_result, mechanics_standardizer, _ = mechanics_kernel(
        source["mechanics_features"],
        source["mechanics_valid_mask"],
        training_indices=training_indices,
        chunk_size=args.chunk_size,
    )
    timings["mechanics_kernel_s"] = time.perf_counter() - start

    start = time.perf_counter()
    pressure_result = pressure_field_kernel(
        source["pressure_channels"],
        source["footprint_overlap_fraction"],
        training_indices=training_indices,
        pressure_weight=0.5,
        gradient_weight=0.5,
        chunk_size=args.chunk_size,
    )
    timings["pressure_kernel_s"] = time.perf_counter() - start

    endpoint_strain = source["mechanics_features"][:, -1, strain_index]
    start = time.perf_counter()
    phase_result, phase_standardizer, _ = endpoint_scalar_kernel(
        endpoint_strain,
        training_indices=training_indices,
        chunk_size=args.chunk_size,
    )
    timings["phase_kernel_s"] = time.perf_counter() - start

    start = time.perf_counter()
    rate_result = loading_rate_kernel(
        source["log_loading_rate"],
        source["specimen_id"],
        training_indices=training_indices,
        chunk_size=args.chunk_size,
    )
    timings["rate_kernel_s"] = time.perf_counter() - start

    start = time.perf_counter()
    composite = composite_kernel(
        mechanics_result.kernel,
        pressure_result.kernel,
        phase_result.kernel,
        rate_result.kernel,
        beta_mechanics=args.beta_mechanics,
        beta_pressure=args.beta_pressure,
        beta_interaction=args.beta_interaction,
        rho=args.rho,
        spatial=None,
    )
    timings["composite_kernel_s"] = time.perf_counter() - start
    timings["total_kernel_s"] = float(sum(timings.values()))

    kernel_map = {
        "K_M": mechanics_result.kernel,
        "K_P": pressure_result.kernel,
        "K_T": phase_result.kernel,
        "K_R": rate_result.kernel,
        "K_M x K_P": mechanics_result.kernel * pressure_result.kernel,
        "K_PI prototype": composite,
    }
    output_npz = args.output_dir / "kernel_matrices.npz"
    output_png = args.output_dir / "kernel_heatmaps.png"
    output_json = args.output_dir / "kernel_diagnostics.json"
    output_csv = args.output_dir / "kernel_diagnostics_table.csv"
    np.savez_compressed(
        output_npz,
        mechanics_kernel=mechanics_result.kernel,
        pressure_kernel=pressure_result.kernel,
        phase_kernel=phase_result.kernel,
        rate_kernel=rate_result.kernel,
        composite_kernel=composite,
        endpoint_pressure_frame=source["endpoint_pressure_frame"],
        endpoint_time_s=source["endpoint_time_s"],
        mechanics_length_scale=np.float64(mechanics_result.length_scale),
        pressure_value_length_scale=np.float64(pressure_result.pressure_length_scale),
        pressure_gradient_length_scale=np.float64(pressure_result.gradient_length_scale),
        phase_length_scale=np.float64(phase_result.length_scale),
        rate_length_scale=np.float64(rate_result.length_scale),
    )
    plot_kernels(kernel_map, source["endpoint_time_s"], specimen_id, output_png)

    diagnostics = {name: kernel_diagnostics(matrix) for name, matrix in kernel_map.items()}
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = ["kernel_name", *next(iter(diagnostics.values())).keys()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, values in diagnostics.items():
            writer.writerow(
                {
                    "kernel_name": name,
                    **{
                        key: json.dumps(value) if isinstance(value, list) else value
                        for key, value in values.items()
                    },
                }
            )
    all_psd = all(item["psd_with_numerical_tolerance"] for item in diagnostics.values())
    n_safe = 6000
    bytes_per_kernel = n_safe * n_safe * np.dtype(DTYPE).itemsize
    summary = {
        "schema_version": "1.0",
        "specimen_id": specimen_id,
        "purpose": "CPU mathematical reference and diagnostic; not a fitted prediction model",
        "sample_count": int(len(training_indices)),
        "dtype": str(np.dtype(DTYPE)),
        "chunk_size": args.chunk_size,
        "training_scope": (
            f"all {len(training_indices)} windows of specimen {specimen_id} for this isolated diagnostic only; "
            "future LOSO runs must refit transformations and scales on training specimens"
        ),
        "definitions": {
            "K_M": "RBF on the complete robust-standardized mechanics history",
            "K_P": (
                "RBF on weighted 32 x 32 pressure histories; pressure-value and gradient "
                "distances use separate training RMS and median-distance scales with equal weights"
            ),
            "K_T": "RBF on endpoint actual axial strain from the first machine record",
            "K_R": "RBF on log representative loading rate",
            "K_S": (
                "set to the all-ones identity factor because no reliable spatial-origin "
                f"label is available ({spatial_origin.get('status', 'not_provided')})"
            ),
            "K_PI_prototype": (
                "K_T * ((1-rho)+rho*K_R) * "
                "(beta_M*K_M + beta_P*K_P + beta_MP*K_M*K_P)"
            ),
        },
        "prototype_weights": {
            "rho": args.rho,
            "beta_mechanics": args.beta_mechanics,
            "beta_pressure": args.beta_pressure,
            "beta_interaction": args.beta_interaction,
            "status": "diagnostic defaults only; tune inside future training folds",
        },
        "fitted_training_scales": {
            "mechanics_feature_median": mechanics_standardizer.median.astype(float).tolist(),
            "mechanics_feature_iqr_or_fallback": mechanics_standardizer.scale.astype(float).tolist(),
            "mechanics_rbf_length_scale": mechanics_result.length_scale,
            "pressure_value_rms_kpa": float(pressure_result.pressure_channel_rms[0]),
            "pressure_gradient_rms_kpa_per_mm": float(pressure_result.pressure_channel_rms[1]),
            "pressure_value_rbf_length_scale": pressure_result.pressure_length_scale,
            "pressure_gradient_rbf_length_scale": pressure_result.gradient_length_scale,
            "phase_median": float(phase_standardizer.median[0]),
            "phase_scale": float(phase_standardizer.scale[0]),
            "phase_rbf_length_scale": phase_result.length_scale,
            "rate_rbf_length_scale": rate_result.length_scale,
            "rate_scale_fit_unit": "unique training specimens, not repeated windows",
        },
        "degeneracy": {
            "mechanics": mechanics_result.degenerate,
            "pressure_value": pressure_result.degenerate_pressure,
            "pressure_gradient": pressure_result.degenerate_gradient,
            "phase": phase_result.degenerate,
            "rate": rate_result.degenerate,
            "rate_interpretation": (
                "expected all-ones kernel for a single specimen; cross-rate behaviour requires multiple specimens"
            ),
        },
        "diagnostics": diagnostics,
        "all_kernels_psd_with_numerical_tolerance": all_psd,
        "off_diagonal_correlations": {
            "K_M_vs_K_P": off_diagonal_correlation(
                mechanics_result.kernel, pressure_result.kernel
            ),
            "K_M_vs_K_T": off_diagonal_correlation(
                mechanics_result.kernel, phase_result.kernel
            ),
            "K_P_vs_K_T": off_diagonal_correlation(
                pressure_result.kernel, phase_result.kernel
            ),
        },
        "timings_seconds": timings,
        "laptop_safe_memory_estimate": {
            "N_train": n_safe,
            "single_float32_kernel_mib": bytes_per_kernel / 1024**2,
            "five_float32_kernels_mib": 5 * bytes_per_kernel / 1024**2,
            "note": "working arrays add memory; cache/release base kernels instead of retaining unnecessary copies",
        },
        "outputs": {
            "kernel_matrices_npz": str(output_npz.resolve()),
            "heatmap_figure": str(output_png.resolve()),
            "diagnostics_csv": str(output_csv.resolve()),
        },
    }
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not all_psd:
        raise RuntimeError("At least one kernel failed the PSD diagnostic")


if __name__ == "__main__":
    main()
