#!/usr/bin/env python3
"""Rate-blind early-contact regime discovery and load/damage associations."""

from __future__ import annotations

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from run_event_stage_bootstrap_analysis import RATE_ORDER
from run_rkhs_evolution_analysis import center_kernel, median_bandwidth, rbf_kernel, robust_fit, robust_transform


EARLY_FEATURES = [
    "pressure_area_ge_50kpa_fraction_of_face",
    "pressure_area_ge_250kpa_fraction_of_face",
    "pressure_centroid_x_fraction",
    "pressure_centroid_y_toward_camera_fraction",
    "pressure_anisotropy",
    "pressure_effective_area_fraction_of_face",
    "pressure_normalized_entropy",
    "pressure_top_10pct_cell_load_fraction",
    "pressure_largest_cluster_fraction_ge_250kpa",
]
PROFILE_FEATURES = {
    "pressure_area_ge_50kpa_fraction_of_face": ("Area ≥50 kPa / face", "%", 100.0),
    "pressure_effective_area_fraction_of_face": ("Effective pressure area / face", "%", 100.0),
    "pressure_normalized_entropy": ("Normalized pressure entropy", "–", 1.0),
    "pressure_top_10pct_cell_load_fraction": ("Top-10% cell load fraction", "%", 100.0),
}
OUTCOME_METRICS = {
    "peak_nominal_stress_mpa": ("Peak nominal stress", "MPa", 1.0),
    "damage_to_peak_force_ratio": ("Visible-damage load / peak load", "–", 1.0),
    "pressure_transition_force_fraction_of_peak": ("RKHS transition load / peak load", "–", 1.0),
    "maximum_visible_damage_area_fraction_pre_cutoff": ("Maximum visible damage area", "%", 100.0),
}
REGIME_COLORS = ["#2563eb", "#d97706", "#7c3aed", "#059669"]
RATE_MARKERS = {"low": "o", "medium": "s", "high": "^"}
MAP_LABEL_OFFSETS = {
    "0.42-2": (7, 10),
    "0.42-3": (7, -13),
    "21-2": (8, 10),
    "210-4": (8, -10),
    "210-1": (-48, -14),
}


def pairwise_rkhs_distance(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    bandwidth = median_bandwidth(vectors)
    kernel, _ = rbf_kernel(vectors, bandwidth)
    distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * kernel))
    return distance, kernel, bandwidth


def kmedoids_exhaustive(
    distance: np.ndarray,
    k: int,
    minimum_cluster_size: int = 1,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    best_cost = float("inf")
    best_labels: np.ndarray | None = None
    best_medoids: tuple[int, ...] | None = None
    for medoids in itertools.combinations(range(len(distance)), k):
        distances_to_medoids = distance[:, medoids]
        labels = np.argmin(distances_to_medoids, axis=1)
        sizes = np.bincount(labels, minlength=k)
        if np.min(sizes) < minimum_cluster_size:
            continue
        cost = float(np.sum(np.min(distances_to_medoids, axis=1)))
        if cost < best_cost - 1e-12:
            best_cost = cost
            best_labels = labels
            best_medoids = medoids
    if best_labels is None or best_medoids is None:
        raise ValueError(f"No k-medoids solution for k={k}, minimum size={minimum_cluster_size}")
    return best_labels, best_medoids, best_cost


def silhouette_samples(distance: np.ndarray, labels: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(labels), dtype=float)
    unique = np.unique(labels)
    for index in range(len(labels)):
        same = np.flatnonzero(labels == labels[index])
        same = same[same != index]
        a = float(np.mean(distance[index, same])) if len(same) else 0.0
        b = min(
            float(np.mean(distance[index, np.flatnonzero(labels == other)]))
            for other in unique
            if other != labels[index]
        )
        scores[index] = (b - a) / max(a, b, 1e-12)
    return scores


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.unique(first)
    second_values = np.unique(second)
    contingency = np.asarray(
        [[np.sum((first == a) & (second == b)) for b in second_values] for a in first_values],
        dtype=int,
    )
    n = len(first)
    if n < 2:
        return 1.0
    sum_pairs = float(sum(comb(int(value), 2) for value in contingency.ravel()))
    row_pairs = float(sum(comb(int(value), 2) for value in contingency.sum(axis=1)))
    column_pairs = float(sum(comb(int(value), 2) for value in contingency.sum(axis=0)))
    total_pairs = float(comb(n, 2))
    expected = row_pairs * column_pairs / total_pairs
    maximum = 0.5 * (row_pairs + column_pairs)
    if maximum <= expected + 1e-12:
        return 1.0 if np.array_equal(first, second) else 0.0
    return (sum_pairs - expected) / (maximum - expected)


def leave_one_out_stability(distance: np.ndarray, full_labels: np.ndarray, k: int) -> np.ndarray:
    scores = []
    for omitted in range(len(distance)):
        keep = np.asarray([index for index in range(len(distance)) if index != omitted])
        sub_distance = distance[np.ix_(keep, keep)]
        try:
            labels, _, _ = kmedoids_exhaustive(sub_distance, k, minimum_cluster_size=2)
        except ValueError:
            labels, _, _ = kmedoids_exhaustive(sub_distance, k, minimum_cluster_size=1)
        scores.append(adjusted_rand_index(full_labels[keep], labels))
    return np.asarray(scores)


def select_cluster_count(distance: np.ndarray) -> tuple[pd.DataFrame, int, dict[int, tuple[np.ndarray, tuple[int, ...], float]]]:
    rows = []
    solutions: dict[int, tuple[np.ndarray, tuple[int, ...], float]] = {}
    for k in [2, 3, 4]:
        labels, medoids, cost = kmedoids_exhaustive(distance, k, minimum_cluster_size=3)
        silhouette = silhouette_samples(distance, labels)
        stability = leave_one_out_stability(distance, labels, k)
        sizes = np.bincount(labels, minlength=k)
        rows.append(
            {
                "cluster_count": k,
                "within_medoid_distance_sum": cost,
                "mean_silhouette": float(np.mean(silhouette)),
                "median_leave_one_out_adjusted_rand": float(np.median(stability)),
                "minimum_leave_one_out_adjusted_rand": float(np.min(stability)),
                "minimum_cluster_size": int(np.min(sizes)),
                "maximum_cluster_size": int(np.max(sizes)),
            }
        )
        solutions[k] = (labels, medoids, cost)
    model_table = pd.DataFrame(rows)
    best = model_table.sort_values(
        ["mean_silhouette", "median_leave_one_out_adjusted_rand", "cluster_count"],
        ascending=[False, False, True],
    ).iloc[0]
    return model_table, int(best["cluster_count"]), solutions


def kernel_pca(kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = center_kernel(kernel)
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    coordinates = vectors[:, :2] * np.sqrt(values[:2])[None, :]
    explained = values[:2] / values.sum() if values.sum() > 0 else np.zeros(2)
    return coordinates, explained


def relabel_by_contact_width(
    labels: np.ndarray,
    trajectories: np.ndarray,
    feature_names: list[str],
    early_indices: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    effective_index = feature_names.index("pressure_effective_area_fraction_of_face")
    cluster_width = {
        cluster: float(np.median(trajectories[labels == cluster][:, early_indices, effective_index]))
        for cluster in np.unique(labels)
    }
    order = sorted(cluster_width, key=cluster_width.get, reverse=True)
    mapping = {old: new for new, old in enumerate(order)}
    return np.asarray([mapping[value] for value in labels]), mapping


def cramer_v(labels: np.ndarray, rates: list[str]) -> float:
    rate_index = {rate: index for index, rate in enumerate(RATE_ORDER)}
    table = np.zeros((len(np.unique(labels)), len(RATE_ORDER)), dtype=float)
    for label, rate in zip(labels, rates):
        table[int(label), rate_index[rate]] += 1.0
    expected = table.sum(axis=1)[:, None] * table.sum(axis=0)[None, :] / table.sum()
    chi2 = float(np.sum((table - expected) ** 2 / np.maximum(expected, 1e-12)))
    return float(np.sqrt(chi2 / (table.sum() * min(table.shape[0] - 1, table.shape[1] - 1))))


def permutation_p_rate(
    labels: np.ndarray,
    rates: list[str],
    rng: np.random.Generator,
    repeats: int,
) -> tuple[float, float]:
    observed = cramer_v(labels, rates)
    rate_array = np.asarray(rates)
    exceed = 0
    for _ in range(repeats):
        exceed += cramer_v(labels, list(rng.permutation(rate_array))) >= observed - 1e-12
    return observed, float((exceed + 1) / (repeats + 1))


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    finite = np.isfinite(values)
    values = values[finite]
    labels = labels[finite]
    if len(values) < 2:
        return float("nan")
    overall = float(np.mean(values))
    total = float(np.sum((values - overall) ** 2))
    if total <= 1e-15:
        return 0.0
    between = sum(
        len(values[labels == cluster])
        * (float(np.mean(values[labels == cluster])) - overall) ** 2
        for cluster in np.unique(labels)
    )
    return float(between / total)


def outcome_associations(
    outcomes: pd.DataFrame,
    assignments: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
) -> pd.DataFrame:
    joined = assignments.merge(outcomes, on=["specimen_id", "rate_group"], how="left")
    rows = []
    for metric in OUTCOME_METRICS:
        values = joined[metric].to_numpy(dtype=float)
        labels = joined["regime_index_1_based"].to_numpy(dtype=int)
        observed = eta_squared(values, labels)
        exceed = 0
        for _ in range(repeats):
            exceed += eta_squared(values, rng.permutation(labels)) >= observed - 1e-12
        rows.append(
            {
                "outcome_metric": metric,
                "eta_squared_contact_regime": observed,
                "permutation_p_value": float((exceed + 1) / (repeats + 1)),
                "permutation_repeats": repeats,
                "finite_specimen_count": int(np.isfinite(values).sum()),
            }
        )
    result = pd.DataFrame(rows)
    order = np.argsort(result["permutation_p_value"].to_numpy(dtype=float))
    q = np.ones(len(result), dtype=float)
    running = 1.0
    for reverse_rank in range(len(result) - 1, -1, -1):
        index = int(order[reverse_rank])
        rank = reverse_rank + 1
        running = min(running, float(result.iloc[index]["permutation_p_value"]) * len(result) / rank)
        q[index] = min(running, 1.0)
    result["bh_q_value_across_outcomes"] = q
    return result


def regime_name(index: int, total: int) -> str:
    if total == 2:
        return ["Broad/distributed", "Localized/concentrated"][index]
    if total == 3:
        return ["Broad/distributed", "Intermediate", "Localized/concentrated"][index]
    return f"Regime {index + 1}"


def plot_regime_map(assignments: pd.DataFrame, explained: np.ndarray, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 7))
    regime_count = assignments["regime_index_1_based"].nunique()
    for row in assignments.itertuples(index=False):
        regime = int(row.regime_index_1_based) - 1
        axis.scatter(
            row.kernel_pc1,
            row.kernel_pc2,
            s=145 if row.is_medoid else 80,
            marker=RATE_MARKERS[row.rate_group],
            facecolor=REGIME_COLORS[regime],
            edgecolor="black" if row.is_medoid else "white",
            linewidth=1.7 if row.is_medoid else 0.7,
            zorder=3,
        )
        short = row.specimen_id.replace("-5-rigid-", "")
        axis.annotate(
            short,
            (row.kernel_pc1, row.kernel_pc2),
            xytext=MAP_LABEL_OFFSETS.get(short, (4, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    regime_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[index], markersize=8,
               label=regime_name(index, regime_count))
        for index in range(regime_count)
    ]
    rate_handles = [
        Line2D([0], [0], marker=RATE_MARKERS[rate], color="black", linestyle="none", markerfacecolor="white",
               markersize=7, label=f"{rate} rate")
        for rate in RATE_ORDER
    ]
    first_legend = axis.legend(handles=regime_handles, loc="upper left", frameon=False, title="Rate-blind regimes")
    axis.add_artist(first_legend)
    axis.legend(handles=rate_handles, loc="lower right", frameon=False, title="Rate shown after clustering")
    axis.set_xlabel(f"Kernel PC1 ({explained[0] * 100:.1f}% kernel eigenvalue)")
    axis.set_ylabel(f"Kernel PC2 ({explained[1] * 100:.1f}% kernel eigenvalue)")
    axis.set_title("Rate-blind early-contact regimes in RKHS geometry")
    axis.grid(alpha=0.18)
    fig.text(0.5, 0.02, "Large black-edged symbols are regime medoids; marker shape encodes loading-rate group only for post hoc interpretation.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_regime_profiles(
    trajectories: np.ndarray,
    grid: np.ndarray,
    labels: np.ndarray,
    output: Path,
) -> None:
    early = np.flatnonzero((grid >= 0.10 - 1e-12) & (grid <= 0.40 + 1e-12))
    regime_count = len(np.unique(labels))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for axis, (feature, (title, unit, scale)) in zip(axes.flat, PROFILE_FEATURES.items()):
        feature_index = EARLY_FEATURES.index(feature)
        for regime in range(regime_count):
            values = trajectories[labels == regime][:, early, feature_index] * scale
            for specimen_values in values:
                axis.plot(grid[early], specimen_values, color=REGIME_COLORS[regime], lw=0.8, alpha=0.20)
            median = np.median(values, axis=0)
            q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
            axis.fill_between(grid[early], q25, q75, color=REGIME_COLORS[regime], alpha=0.15)
            axis.plot(
                grid[early],
                median,
                color=REGIME_COLORS[regime],
                lw=2.2,
                marker="o",
                ms=4,
                label=f"{regime_name(regime, regime_count)} (n={len(values)})",
            )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(alpha=0.20)
        axis.set_xlabel("Machine force / specimen peak force")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=regime_count, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Early pressure evolution within rate-blind contact regimes", fontsize=16, y=0.995)
    fig.text(0.5, 0.900, "Thin lines are specimens; thick lines are medians; shaded bands are interquartile ranges.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.855, hspace=0.30, wspace=0.22)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def plot_regime_outcomes(
    assignments: pd.DataFrame,
    outcomes: pd.DataFrame,
    associations: pd.DataFrame,
    output: Path,
) -> None:
    joined = assignments.merge(outcomes, on=["specimen_id", "rate_group"], how="left")
    regime_count = assignments["regime_index_1_based"].nunique()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    rng = np.random.default_rng(20261002)
    for axis, (metric, (title, unit, scale)) in zip(axes.flat, OUTCOME_METRICS.items()):
        association = associations[associations["outcome_metric"] == metric].iloc[0]
        for regime in range(regime_count):
            values = joined[joined["regime_index_1_based"] == regime + 1][metric].to_numpy(dtype=float) * scale
            values = values[np.isfinite(values)]
            x = float(regime)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            axis.scatter(np.full(len(values), x) + jitter, values, color=REGIME_COLORS[regime], s=38, alpha=0.70,
                         edgecolors="white", linewidths=0.5, zorder=3)
            if len(values):
                q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
                axis.add_patch(Rectangle((x - 0.16, q25), 0.32, q75 - q25, facecolor=REGIME_COLORS[regime], alpha=0.18, edgecolor="none"))
                axis.plot([x - 0.16, x + 0.16], [median, median], color=REGIME_COLORS[regime], lw=2.4)
        axis.set_title(f"{title}\nη²={association['eta_squared_contact_regime']:.2f}, q={association['bh_q_value_across_outcomes']:.3f}")
        axis.set_ylabel(unit)
        axis.set_xticks(np.arange(regime_count), [regime_name(index, regime_count) for index in range(regime_count)], rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.20)
    fig.suptitle("Mechanical and visible-damage outcomes by early contact regime", fontsize=16, y=0.99)
    fig.text(0.5, 0.945, "Points are specimens; boxes show interquartile ranges and horizontal bars show medians. No long confidence whiskers are used.", ha="center", fontsize=10)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.88, hspace=0.36, wspace=0.24)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/early_contact_regimes"),
    )
    parser.add_argument("--permutation-repeats", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    balanced = pd.read_csv(args.root / "analysis" / "balanced_loading_trajectories.csv")
    specimen_ids = list(dict.fromkeys(balanced["specimen_id"].astype(str)))
    grid = np.sort(balanced["force_fraction_of_specimen_peak"].unique().astype(float))
    rates: list[str] = []
    trajectories = []
    for specimen in specimen_ids:
        rows = balanced[balanced["specimen_id"] == specimen].sort_values("force_fraction_of_specimen_peak")
        rates.append(str(rows.iloc[0]["rate_group"]))
        trajectories.append(rows[EARLY_FEATURES].to_numpy(dtype=float))
    trajectories = np.asarray(trajectories)
    early = np.flatnonzero((grid >= 0.10 - 1e-12) & (grid <= 0.40 + 1e-12))
    early_states = trajectories[:, early, :]
    center, scale = robust_fit(early_states.reshape(-1, early_states.shape[-1]))
    standardized = robust_transform(
        early_states.reshape(-1, early_states.shape[-1]), center, scale
    ).reshape(early_states.shape)
    vectors = standardized.reshape(len(specimen_ids), -1)
    distance, kernel, bandwidth = pairwise_rkhs_distance(vectors)
    model_table, selected_k, solutions = select_cluster_count(distance)
    labels, medoids, _ = solutions[selected_k]
    labels, mapping = relabel_by_contact_width(labels, trajectories, EARLY_FEATURES, early)
    medoids = tuple(medoids[old] for old, _ in sorted(mapping.items(), key=lambda item: item[1]))
    silhouette = silhouette_samples(distance, labels)
    coordinates, explained = kernel_pca(kernel)

    assignments = pd.DataFrame(
        {
            "specimen_id": specimen_ids,
            "rate_group": rates,
            "regime_index_1_based": labels + 1,
            "regime_name": [regime_name(int(label), selected_k) for label in labels],
            "is_medoid": [int(index in medoids) for index in range(len(specimen_ids))],
            "silhouette_sample": silhouette,
            "kernel_pc1": coordinates[:, 0],
            "kernel_pc2": coordinates[:, 1],
        }
    )
    rng = np.random.default_rng(args.seed)
    rate_v, rate_p = permutation_p_rate(labels, rates, rng, args.permutation_repeats)
    outcomes = pd.read_csv(args.root / "academic" / "ice_load_specimen_metrics.csv")
    outcome_table = outcome_associations(outcomes, assignments, rng, args.permutation_repeats)

    model_table.to_csv(args.output / "contact_regime_model_selection.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(args.output / "early_contact_regime_assignments.csv", index=False, encoding="utf-8-sig")
    outcome_table.to_csv(args.output / "contact_regime_outcome_associations.csv", index=False, encoding="utf-8-sig")
    association = pd.DataFrame(
        [
            {
                "selected_cluster_count": selected_k,
                "cramers_v_contact_regime_vs_rate_group": rate_v,
                "permutation_p_value": rate_p,
                "permutation_repeats": args.permutation_repeats,
                "rate_labels_used_during_clustering": 0,
            }
        ]
    )
    association.to_csv(args.output / "contact_regime_rate_association.csv", index=False, encoding="utf-8-sig")
    plot_regime_map(assignments, explained, args.output / "early_contact_regime_map.png")
    plot_regime_profiles(trajectories, grid, labels, args.output / "early_contact_regime_profiles_iqr.png")
    plot_regime_outcomes(assignments, outcomes, outcome_table, args.output / "early_contact_regime_outcomes_iqr.png")

    selected = model_table[model_table["cluster_count"] == selected_k].iloc[0]
    lines = [
        "# 早期接触制度的速率盲法聚类摘要",
        "",
        f"- 聚类仅使用 10%–40% 峰值载荷区间的 {len(EARLY_FEATURES)} 个压力拓扑特征，不向算法提供加载速率、峰值载荷或破坏标签。",
        f"- 在 k=2–4 且每类至少3个试件的预设候选中，按平均轮廓系数优先选出 k={selected_k}。",
        f"- 选定模型平均轮廓系数为 {selected['mean_silhouette']:.3f}，删一试件调整Rand指数中位数为 {selected['median_leave_one_out_adjusted_rand']:.3f}。",
        f"- 聚类完成后再检验速率关联：Cramér's V={rate_v:.3f}，固定次数置换 p={rate_p:.5f}。",
        "- 制度编号按有效承压面积由宽到窄重新排序，仅用于解释；不改变聚类成员。",
        "",
        "## 解释边界",
        "",
        "1. 速率标签未参与聚类，因此聚类—速率关联不是由监督分类直接制造；但速率仍可能与批次、端面状态和摆放共同混杂。",
        "2. n=16 只支持探索性制度识别；若某一制度仅有3个试件，不能用平滑小提琴分布或正态误差模型营造高精度印象。",
        "3. 主图采用全部试件、组中位数和四分位区间；置换p值与效应量表保留完整不确定性。",
        "4. 接触制度与峰值应力、损伤载荷或可见损伤面积的关联不等于接触制度导致破坏结果。",
    ]
    for row in outcome_table.itertuples(index=False):
        lines.append(
            f"- {OUTCOME_METRICS[row.outcome_metric][0]}：η²={row.eta_squared_contact_regime:.3f}，"
            f"p={row.permutation_p_value:.5f}，q={row.bh_q_value_across_outcomes:.5f}。"
        )
    (args.output / "early_contact_regime_summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "included_specimens": len(specimen_ids),
        "early_force_fraction_range": [0.10, 0.40],
        "features": EARLY_FEATURES,
        "rate_blind_clustering": True,
        "candidate_cluster_counts": [2, 3, 4],
        "minimum_cluster_size": 3,
        "selected_cluster_count": selected_k,
        "rbf_bandwidth": bandwidth,
        "permutation_repeats": args.permutation_repeats,
        "random_seed": args.seed,
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
    }
    (args.output / "early_contact_regime_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
