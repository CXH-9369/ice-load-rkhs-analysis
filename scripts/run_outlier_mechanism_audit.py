#!/usr/bin/env python3
"""Trace pressure-topology outliers through load, pressure, video and quality evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from create_multimodal_event_dashboards import (
    STAGE_COLORS,
    as_float,
    first_crossing,
    specimen_assets,
)
from run_early_contact_regime_analysis import REGIME_COLORS


CHANGE_METRICS = {
    "pressure_area_ge_50kpa_fraction_of_face": "Area >=50 kPa / face",
    "pressure_effective_area_fraction_of_face": "Effective pressure area / face",
    "pressure_normalized_entropy": "Normalized pressure entropy",
    "pressure_top_10pct_cell_load_fraction": "Top-10% cell load fraction",
}
RATE_MARKERS = {"low": "o", "medium": "s", "high": "^"}
SHORT_CHANGE_LABELS = {
    "pressure_area_ge_50kpa_fraction_of_face": "A50",
    "pressure_effective_area_fraction_of_face": "Aeff",
    "pressure_normalized_entropy": "Entropy",
    "pressure_top_10pct_cell_load_fraction": "Top10",
}
MANUAL_MULTIMODAL_INTERPRETATIONS = {
    "-5-rigid-0.42-1": (
        "contact_expansion_without_large_visible_damage",
        "Contact support expands from the camera-side half toward most of the face, while the retained visible-damage area remains very small; interpret primarily as boundary seating/contact redistribution, not a fracture predictor.",
    ),
    "-5-rigid-0.42-3": (
        "contact_expansion_before_near_peak_visible_damage",
        "Broad contact expansion and load deconcentration precede a localized visible side-surface response near peak load; this is a useful mechanism example but not proof of causation.",
    ),
    "-5-rigid-21-1": (
        "real_visible_damage_pressure_amplitude_measurement_limited",
        "The side-surface damage is clear, but the damage-state film/machine force ratio and saturated cells make absolute pressure amplitude unreliable; normalized spatial redistribution remains descriptive only.",
    ),
    "-5-rigid-0.42-2": (
        "event_order_and_pressure_timing_exception",
        "Visible damage occurs well before the RKHS transition, and the nearest damage pressure frame is offset by about 0.545 s; this specimen is a counterexample to a universal precursor narrative.",
    ),
    "-5-rigid-210-4": (
        "clear_event_order_exception",
        "Visible damage occurs substantially before the strongest RKHS transition; the later pressure redistribution cannot be described as a precursor to the already observed damage.",
    ),
    "-5-rigid-21-3": (
        "normalized_redistribution_force_consistency_limited",
        "The normalized pressure map contracts and becomes more concentrated, but the low film/machine force ratio limits absolute-load interpretation.",
    ),
    "-5-rigid-210-2": (
        "near_coincident_events_pressure_resolution_limited",
        "Visible damage precedes the RKHS transition by only one pressure frame; event ordering is resolved by the high-speed record but the pressure-state difference is near the film's temporal resolution.",
    ),
    "-5-rigid-21-4": (
        "visible_damage_pressure_topology_decoupling",
        "A large visible side-surface damage area is present while the normalized contact-pressure topology changes by only about one percentage point; contact pressure alone misses this damage development.",
    ),
    "-5-rigid-210-3": (
        "same_pressure_frame_state_collapse",
        "The RKHS-transition and visible-damage events map to the same pressure frame, so their pressure fields are identical and no transition-to-damage pressure change can be resolved.",
    ),
}


def robust_topology_score(changes: pd.DataFrame) -> np.ndarray:
    columns = [f"damage_minus_transition__{metric}" for metric in CHANGE_METRICS]
    values = changes[columns].to_numpy(dtype=float)
    center = np.nanmedian(values, axis=0)
    q25, q75 = np.nanpercentile(values, [25, 75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
    return np.sqrt(np.sum(((values - center) / scale) ** 2, axis=1))


def build_audit_table(
    changes: pd.DataFrame,
    assignments: pd.DataFrame,
    metrics: pd.DataFrame,
    stages: pd.DataFrame,
) -> pd.DataFrame:
    table = changes.merge(
        assignments[["specimen_id", "rate_group", "regime_index_1_based", "regime_name"]],
        on=["specimen_id", "rate_group", "regime_index_1_based", "regime_name"],
        validate="one_to_one",
    ).merge(metrics, on=["specimen_id", "rate_group"], validate="one_to_one")
    event_stages = stages[stages["stage"].isin(["rkhs_transition", "visible_damage"])].copy()
    event_stages["absolute_pressure_machine_time_lag_s"] = np.abs(
        event_stages["pressure_frame_time_s"].to_numpy(dtype=float)
        - event_stages["machine_time_s"].to_numpy(dtype=float)
    )
    maximum_lag = (
        event_stages.groupby("specimen_id")["absolute_pressure_machine_time_lag_s"]
        .max()
        .rename("maximum_event_pressure_machine_time_lag_s")
    )
    table = table.merge(maximum_lag, left_on="specimen_id", right_index=True, how="left")
    frame_pivot = event_stages.pivot(
        index="specimen_id", columns="stage", values="pressure_frame_number"
    ).astype(int)
    frame_separation = np.abs(
        frame_pivot["visible_damage"] - frame_pivot["rkhs_transition"]
    ).rename("transition_damage_pressure_frame_separation")
    table = table.merge(frame_separation, left_on="specimen_id", right_index=True, how="left")
    table["pressure_resolution_limited_flag"] = (
        table["transition_damage_pressure_frame_separation"] <= 1
    ).astype(int)
    table["topology_change_robust_magnitude"] = robust_topology_score(table)
    table["topology_change_rank"] = table["topology_change_robust_magnitude"].rank(
        ascending=False, method="min"
    ).astype(int)
    table["visible_damage_area_rank"] = table[
        "maximum_visible_damage_area_fraction_pre_cutoff"
    ].rank(ascending=False, method="min").astype(int)
    table["temporal_reversal_flag"] = (table["pressure_transition_precedes_damage"] == 0).astype(int)
    table["measurement_limitation_flag"] = (
        (table["damage_film_to_machine_force_ratio"] > 2.0)
        | (table["damage_film_to_machine_force_ratio"] < 0.15)
        | (table["damage_footprint_capture_fraction"] < 0.90)
        | (table["maximum_event_pressure_machine_time_lag_s"] > 0.05)
        | (table["pressure_resolution_limited_flag"] == 1)
    ).astype(int)
    table["saturation_flag"] = (
        (table["damage_saturated_cell_count"] > 0) | (table["peak_saturated_cell_count"] > 0)
    ).astype(int)
    reasons: list[str] = []
    for row in table.itertuples(index=False):
        row_reasons = []
        if row.topology_change_rank <= 3:
            row_reasons.append("top-3 pressure-topology change")
        if row.temporal_reversal_flag:
            row_reasons.append("visible damage precedes RKHS transition")
        if row.measurement_limitation_flag:
            row_reasons.append("measurement/synchronization limitation")
        if row.pressure_resolution_limited_flag:
            row_reasons.append("transition and damage within one pressure frame")
        if row.visible_damage_area_rank <= 2:
            row_reasons.append("top-2 visible-damage area")
        reasons.append("; ".join(row_reasons))
    table["case_review_reasons"] = reasons
    table["selected_for_case_review"] = (table["case_review_reasons"] != "").astype(int)
    return table.sort_values(
        ["selected_for_case_review", "topology_change_robust_magnitude"], ascending=[False, False]
    )


def plot_audit_overview(table: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
    for row in table.itertuples(index=False):
        regime = int(row.regime_index_1_based)
        edge = "#b91c1c" if row.temporal_reversal_flag else "#111827"
        size = 58 + min(int(row.damage_saturated_cell_count), 20) * 3
        axes[0].scatter(
            row.topology_change_robust_magnitude,
            100.0 * row.maximum_visible_damage_area_fraction_pre_cutoff,
            marker=RATE_MARKERS[row.rate_group],
            s=size,
            facecolor=REGIME_COLORS[regime - 1],
            edgecolor=edge,
            linewidth=1.5,
            alpha=0.82,
        )
        axes[1].scatter(
            row.maximum_event_pressure_machine_time_lag_s,
            row.damage_film_to_machine_force_ratio,
            marker=RATE_MARKERS[row.rate_group],
            s=size,
            facecolor=REGIME_COLORS[regime - 1],
            edgecolor=edge,
            linewidth=1.5,
            alpha=0.82,
        )
        if row.selected_for_case_review:
            short = row.specimen_id.replace("-5-rigid-", "")
            axes[0].annotate(short, (row.topology_change_robust_magnitude, 100.0 * row.maximum_visible_damage_area_fraction_pre_cutoff), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if row.measurement_limitation_flag:
            short = row.specimen_id.replace("-5-rigid-", "")
            axes[1].annotate(short, (row.maximum_event_pressure_machine_time_lag_s, row.damage_film_to_machine_force_ratio), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[0].set_xlabel("Robust magnitude of transition-to-damage topology difference")
    axes[0].set_ylabel("Maximum visible damage area before cutoff (%)")
    axes[0].set_title("Physical-response priority")
    axes[1].set_xlabel("Maximum event pressure-machine time mismatch (s)")
    axes[1].set_ylabel("Film force / machine force at visible-damage state")
    axes[1].set_yscale("log")
    axes[1].axhspan(0.15, 2.0, color="#6b7280", alpha=0.08)
    axes[1].set_title("Measurement-quality context")
    for axis in axes:
        axis.grid(alpha=0.18)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[0], markeredgecolor="#111827", label="Broad/distributed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[1], markeredgecolor="#111827", label="Localized/concentrated"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#b91c1c", markeredgewidth=1.5, label="Damage precedes transition"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Multimodal audit priorities for pressure-topology outliers", fontsize=16, y=0.995)
    fig.text(0.5, 0.02, "Marker shape denotes loading-rate group; marker size increases with saturated-cell count. Labels identify cases selected for detailed review.", ha="center")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.14, wspace=0.24)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def normalized_load_history(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, float, float]:
    time = np.asarray([as_float(row["time_s"]) for row in rows])
    force = np.asarray([as_float(row["force_fraction_of_specimen_peak"]) for row in rows])
    time10 = as_float(first_crossing(rows, 0.10)["time_s"])
    peak_index = int(np.nanargmax(force))
    peak_time = time[peak_index]
    return (time - time10) / (peak_time - time10), force, time10, peak_time


def plot_case_plate(
    workspace: Path,
    specimen: str,
    audit_row: pd.Series,
    rows: list[dict[str, str]],
    metric: dict[str, str],
    output: Path,
) -> None:
    stages, _, overlap, capture, _, _, _ = specimen_assets(workspace, specimen, rows, metric)
    transition = stages[1]
    damage = stages[2]
    figure = plt.figure(figsize=(17.5, 10.5))
    grid = GridSpec(2, 4, figure=figure, height_ratios=[1.0, 1.05], hspace=0.25, wspace=0.20)
    pressure_states = [transition["pressure_percent"], damage["pressure_percent"]]
    vmax = float(np.percentile(np.asarray(pressure_states), 99.5))
    image = None
    for column, (stage, title) in enumerate([(transition, "RKHS transition"), (damage, "Visible-damage state")]):
        axis = figure.add_subplot(grid[0, column])
        image = axis.imshow(stage["pressure_percent"], origin="lower", extent=[-35, 35, -35, 35], cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
        saturated = np.asarray(stage["pressure_raw"]) >= 12880.0 - 1e-3
        if np.any(saturated):
            axis.contour(np.linspace(-34, 34, 32), np.linspace(-34, 34, 32), saturated.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        axis.set_title(f"{title}\npressure load share")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y toward camera (mm)")
        axis.set_xticks([-35, 0, 35])
        axis.set_yticks([-35, 0, 35])
    difference = np.asarray(damage["pressure_percent"]) - np.asarray(transition["pressure_percent"])
    limit = max(float(np.percentile(np.abs(difference), 99.0)), 1e-6)
    diff_axis = figure.add_subplot(grid[0, 2])
    diff_image = diff_axis.imshow(difference, origin="lower", extent=[-35, 35, -35, 35], cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    diff_axis.set_title("Semantic state difference\ndamage minus transition")
    diff_axis.set_xlabel("x (mm)")
    diff_axis.set_ylabel("y toward camera (mm)")
    diff_axis.set_xticks([-35, 0, 35])
    diff_axis.set_yticks([-35, 0, 35])
    change_axis = figure.add_subplot(grid[0, 3])
    metric_names = list(CHANGE_METRICS)
    values = np.asarray(
        [audit_row[f"damage_minus_transition__{name}"] for name in metric_names], dtype=float
    ) * 100.0
    colors = ["#2563eb" if value >= 0 else "#dc2626" for value in values]
    positions = np.arange(len(values))
    change_axis.barh(positions, values, color=colors, alpha=0.78)
    change_axis.axvline(0.0, color="#111827", lw=1.0)
    change_axis.set_yticks(positions)
    change_axis.set_yticklabels([SHORT_CHANGE_LABELS[name] for name in metric_names])
    change_axis.invert_yaxis()
    change_axis.set_xlabel("Damage minus transition (percentage points)")
    change_axis.set_title("Pressure-topology changes")
    change_axis.grid(axis="x", alpha=0.18)

    for column, (stage, title) in enumerate([(transition, "Side view at RKHS transition"), (damage, "Side view at visible-damage event")]):
        axis = figure.add_subplot(grid[1, column])
        if stage["video_image"] is None:
            axis.set_facecolor("#eeeeee")
            axis.text(0.5, 0.5, "High-speed frame not recorded", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.imshow(stage["video_image"])
        axis.set_title(f"{title}\nframe {stage['video_frame_number']}")
        axis.axis("off")

    load_axis = figure.add_subplot(grid[1, 2:])
    normalized_time, force_fraction, time10, peak_time = normalized_load_history(rows)
    load_axis.plot(normalized_time, force_fraction, color="#111827", lw=1.6)
    for stage, color, label in [
        (transition, STAGE_COLORS[1], "RKHS transition"),
        (damage, STAGE_COLORS[2], "Visible damage"),
    ]:
        event_x = (float(stage["machine_time_s"]) - time10) / (peak_time - time10)
        load_axis.axvline(event_x, color=color, lw=1.7, label=label)
    load_axis.axvline(1.0, color="#6b7280", lw=1.1, ls="--", label="Machine peak")
    transition_x = (float(transition["machine_time_s"]) - time10) / (peak_time - time10)
    damage_x = (float(damage["machine_time_s"]) - time10) / (peak_time - time10)
    load_axis.set_xlim(min(-0.20, transition_x - 0.12, damage_x - 0.12), max(1.15, transition_x + 0.12, damage_x + 0.12))
    load_axis.set_xlabel("Normalized time: 10% load = 0, machine peak = 1")
    load_axis.set_ylabel("Machine force / specimen peak force")
    load_axis.set_title("Load history and event order")
    load_axis.grid(alpha=0.18)
    load_axis.legend(frameon=False, loc="best")
    quality = (
        f"Review reasons: {audit_row['case_review_reasons']}\n"
        f"event pressure-machine mismatch <= {audit_row['maximum_event_pressure_machine_time_lag_s']:.4f} s | "
        f"film/machine at damage = {audit_row['damage_film_to_machine_force_ratio']:.3f}\n"
        f"damage saturated cells = {int(audit_row['damage_saturated_cell_count'])} | "
        f"footprint capture = {audit_row['damage_footprint_capture_fraction']:.3f} | "
        f"max visible damage area = {100.0 * audit_row['maximum_visible_damage_area_fraction_pre_cutoff']:.1f}%"
    )
    figure.text(0.5, 0.035, quality, ha="center", va="bottom", fontsize=10)
    order = "VISIBLE DAMAGE PRECEDES RKHS TRANSITION" if audit_row["temporal_reversal_flag"] else "RKHS transition precedes visible damage"
    figure.suptitle(
        f"{specimen} | {audit_row['regime_name']} | {order}\n"
        "pressure-map lower edge faces the camera; cyan contours denote saturated cells",
        fontsize=15,
        y=0.995,
        color="#b91c1c" if audit_row["temporal_reversal_flag"] else "#111827",
    )
    if image is not None:
        cax = figure.add_axes([0.475, 0.58, 0.010, 0.22])
        cb = figure.colorbar(image, cax=cax)
        cb.set_label("Cell share of contact load (%)")
    cax_diff = figure.add_axes([0.685, 0.58, 0.010, 0.22])
    cb_diff = figure.colorbar(diff_image, cax=cax_diff)
    cb_diff.set_label("Change in cell load share (percentage points)")
    figure.subplots_adjust(left=0.055, right=0.96, top=0.89, bottom=0.14)
    figure.savefig(output, dpi=210)
    plt.close(figure)
    capture.release()


def write_summary(table: pd.DataFrame, output: Path) -> None:
    selected = table[table["selected_for_case_review"] == 1]
    lines = [
        "# 压力拓扑离群试件的多模态溯源摘要",
        "",
        "- 离群排序使用四项‘可见损伤状态减 RKHS 转变状态’压力拓扑差的全样本稳健标准化欧氏幅值；它是复核优先级，不是物理损伤评分或预测分数。",
        "- 详细复核集合取四类证据的并集：压力拓扑变化前三、可见损伤早于 RKHS 转变、明确测量/同步限制、可见损伤面积前二。",
        "- 压力差图使用每个状态内部按接触载荷归一化后的单元载荷份额，因此显示载荷空间重分配，而不是绝对压力增量。",
        "",
        f"共选出 {len(selected)} 个试件进入详细多模态复核：",
        "",
    ]
    for row in selected.itertuples(index=False):
        interpretation = MANUAL_MULTIMODAL_INTERPRETATIONS.get(row.specimen_id, ("not_reviewed", ""))
        lines.append(
            f"- `{row.specimen_id}`：拓扑变化排序 {row.topology_change_rank}/16，"
            f"可见损伤面积排序 {row.visible_damage_area_rank}/16；{row.case_review_reasons}。"
            f"人工多模态判读：`{interpretation[0]}`——{interpretation[1]}"
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "1. 若可见损伤早于 RKHS 转变，该试件不能作为‘RKHS 转变预示破坏’的正例。",
            "2. 薄膜合力/机器载荷异常、事件压力帧时间错配或明显饱和只构成测量限制；不能据此否定高速画面中真实发生的破坏。",
            "3. 高速画面与压力场同时出现空间突变时，可称为多模态共现；在没有独立因果操控时仍不能写成压力重排导致裂纹。",
            "4. 分形维数继续只在人工确认损伤后且自动质量门控有效的帧内解释，不由本离群评分补值。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/outlier_mechanism_audit"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    case_dir = args.output / "case_plates"
    case_dir.mkdir(parents=True, exist_ok=True)

    academic = args.root / "academic"
    changes = pd.read_csv(academic / "contact_regime_conditioned" / "transition_to_damage_pressure_changes.csv")
    assignments = pd.read_csv(academic / "early_contact_regimes" / "early_contact_regime_assignments.csv")
    metrics = pd.read_csv(academic / "ice_load_specimen_metrics.csv")
    stages = pd.read_csv(academic / "bootstrap_event_analysis" / "event_stage_pressure_metrics.csv")
    audit = build_audit_table(changes, assignments, metrics, stages)
    audit.to_csv(args.output / "outlier_mechanism_audit_table.csv", index=False, encoding="utf-8-sig")
    plot_audit_overview(audit, args.output / "outlier_mechanism_audit_overview.png")

    state_table = pd.read_csv(args.root / "dataset" / "evolution_state_table.csv", dtype=str)
    metric_by_id = {row["specimen_id"]: row for row in metrics.astype(str).to_dict("records")}
    selected_ids = list(audit.loc[audit["selected_for_case_review"] == 1, "specimen_id"])
    manual_rows = [
        {
            "specimen_id": specimen,
            "manual_multimodal_class": MANUAL_MULTIMODAL_INTERPRETATIONS[specimen][0],
            "manual_interpretation": MANUAL_MULTIMODAL_INTERPRETATIONS[specimen][1],
            "interpretation_scope": "descriptive_not_causal",
        }
        for specimen in selected_ids
    ]
    pd.DataFrame(manual_rows).to_csv(
        args.output / "manual_multimodal_case_interpretations.csv", index=False, encoding="utf-8-sig"
    )
    for specimen in selected_ids:
        rows = state_table[state_table["specimen_id"] == specimen].to_dict("records")
        audit_row = audit[audit["specimen_id"] == specimen].iloc[0]
        plot_case_plate(
            args.workspace,
            specimen,
            audit_row,
            rows,
            metric_by_id[specimen],
            case_dir / f"{specimen}_outlier_case_plate.png",
        )
    write_summary(audit, args.output / "outlier_mechanism_audit_summary_zh.md")
    metadata = {
        "included_specimens": len(audit),
        "selected_case_count": len(selected_ids),
        "selected_specimens": selected_ids,
        "selection_policy": [
            "top_3_robust_pressure_topology_change",
            "visible_damage_precedes_rkhs_transition",
            "measurement_or_synchronization_limitation",
            "top_2_visible_damage_area",
        ],
        "pressure_orientation": "lower_edge_faces_high_speed_camera_no_left_right_mirror",
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
    }
    (args.output / "outlier_mechanism_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
