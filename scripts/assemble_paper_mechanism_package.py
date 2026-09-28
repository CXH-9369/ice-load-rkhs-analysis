#!/usr/bin/env python3
"""Assemble a paper-facing mechanism figure, caption catalog and claim guardrails."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from run_early_contact_regime_analysis import REGIME_COLORS
from run_event_stage_bootstrap_analysis import RATE_COLORS


RATE_MARKERS = {"low": "o", "medium": "s", "high": "^"}
REGIME_NAMES = {1: "Broad/distributed", 2: "Localized/concentrated"}


def plot_regime_geometry(axis: plt.Axes, assignments: pd.DataFrame) -> None:
    for row in assignments.itertuples(index=False):
        regime = int(row.regime_index_1_based)
        axis.scatter(
            row.kernel_pc1,
            row.kernel_pc2,
            marker=RATE_MARKERS[row.rate_group],
            s=82 if row.is_medoid else 52,
            facecolor=REGIME_COLORS[regime - 1],
            edgecolor="#111827",
            linewidth=1.6 if row.is_medoid else 0.7,
            alpha=0.88,
        )
    axis.set_xlabel("Kernel PC1")
    axis.set_ylabel("Kernel PC2")
    axis.set_title("a  Rate-blind early-contact geometry")
    axis.grid(alpha=0.18)
    axis.text(
        0.98,
        0.03,
        "k=2; silhouette=0.326\nregime vs rate: V=0.856, p=0.0055",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def plot_post40_evolution(
    axis: plt.Axes,
    distances: pd.DataFrame,
    conditioned_mmd: pd.DataFrame,
) -> None:
    for rate in ["medium", "high"]:
        subset = distances[distances["rate_group"] == rate]
        for _, rows in subset.groupby("specimen_id"):
            rows = rows.sort_values("force_fraction_of_specimen_peak")
            axis.plot(
                rows["force_fraction_of_specimen_peak"],
                rows["rkhs_distance_from_40pct_load_state"],
                color=RATE_COLORS[rate],
                lw=0.8,
                alpha=0.24,
            )
        grouped = subset.groupby("force_fraction_of_specimen_peak")[
            "rkhs_distance_from_40pct_load_state"
        ]
        x, medians, q25s, q75s = [], [], [], []
        for force_fraction, values in grouped:
            q25, median, q75 = np.percentile(values.to_numpy(dtype=float), [25, 50, 75])
            x.append(float(force_fraction))
            medians.append(median)
            q25s.append(q25)
            q75s.append(q75)
        axis.fill_between(x, q25s, q75s, color=RATE_COLORS[rate], alpha=0.14)
        axis.plot(
            x,
            medians,
            color=RATE_COLORS[rate],
            lw=2.2,
            marker="o",
            ms=3.5,
            label=f"{rate.title()} (n={subset['specimen_id'].nunique()})",
        )
    result = conditioned_mmd[
        conditioned_mmd["trajectory_definition"] == "increment_from_40pct"
    ].iloc[0]
    axis.set_xlabel("Machine force / specimen peak force")
    axis.set_ylabel("RKHS distance from own 40%-load state")
    axis.set_title("b  Later evolution within the localized-contact regime")
    axis.grid(alpha=0.18)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.98,
        0.04,
        f"50%-100% adjusted trajectory\nMMD²={result.mmd2:.3f}, exact p={result.exact_permutation_p_value:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def plot_decoupling(axis: plt.Axes, audit: pd.DataFrame) -> None:
    labels = {"-5-rigid-0.42-1", "-5-rigid-21-4", "-5-rigid-0.42-2", "-5-rigid-210-4"}
    offsets = {
        "-5-rigid-0.42-1": (-44, 6),
        "-5-rigid-21-4": (5, 5),
        "-5-rigid-0.42-2": (5, 5),
        "-5-rigid-210-4": (5, -12),
    }
    for row in audit.itertuples(index=False):
        regime = int(row.regime_index_1_based)
        edge = "#b91c1c" if row.temporal_reversal_flag else "#111827"
        axis.scatter(
            row.topology_change_robust_magnitude,
            100.0 * row.maximum_visible_damage_area_fraction_pre_cutoff,
            marker=RATE_MARKERS[row.rate_group],
            s=58,
            facecolor=REGIME_COLORS[regime - 1],
            edgecolor=edge,
            linewidth=1.4,
            alpha=0.86,
        )
        if row.specimen_id in labels:
            axis.annotate(
                row.specimen_id.replace("-5-rigid-", ""),
                (
                    row.topology_change_robust_magnitude,
                    100.0 * row.maximum_visible_damage_area_fraction_pre_cutoff,
                ),
                xytext=offsets[row.specimen_id],
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Robust pressure-topology change magnitude")
    axis.set_ylabel("Maximum visible damage area before cutoff (%)")
    axis.set_title("c  Interface-pressure and side-damage decoupling")
    axis.grid(alpha=0.18)
    axis.text(
        0.98,
        0.96,
        "Two decoupled extremes:\n"
        "0.42-1: large contact redistribution, little visible damage\n"
        "21-4: large visible damage, nearly stable contact topology",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )


def plot_event_order(axis: plt.Axes, events: pd.DataFrame) -> None:
    ordered = events.sort_values(
        ["pressure_transition_precedes_damage", "regime_index_1_based", "pressure_transition_time_normalized_10pct_to_peak"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    for position, row in ordered.iterrows():
        transition = float(row["pressure_transition_time_normalized_10pct_to_peak"])
        damage = float(row["damage_time_normalized_10pct_to_peak"])
        precedes = int(row["pressure_transition_precedes_damage"]) == 1
        color = "#2563eb" if precedes else "#b91c1c"
        axis.plot([transition, damage], [position, position], color=color, lw=1.6, alpha=0.72)
        axis.scatter(transition, position, marker="o", s=26, color="#111827", zorder=3)
        axis.scatter(damage, position, marker="D", s=28, color=color, zorder=3)
    axis.axvline(1.0, color="#6b7280", lw=1.0, ls="--")
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels([value.replace("-5-rigid-", "") for value in ordered["specimen_id"]], fontsize=7.5)
    axis.invert_yaxis()
    axis.set_xlabel("Normalized time: 10% load = 0, machine peak = 1")
    axis.set_ylabel("Specimen")
    axis.set_title("d  Event order is not universal")
    axis.grid(axis="x", alpha=0.18)
    axis.text(
        0.98,
        0.04,
        "RKHS transition precedes visible damage: 12/16\nreverse order: 4/16",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def plot_framework(
    assignments: pd.DataFrame,
    distances: pd.DataFrame,
    conditioned_mmd: pd.DataFrame,
    audit: pd.DataFrame,
    events: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5))
    plot_regime_geometry(axes[0, 0], assignments)
    plot_post40_evolution(axes[0, 1], distances, conditioned_mmd)
    plot_decoupling(axes[1, 0], audit)
    plot_event_order(axes[1, 1], events)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[0], markeredgecolor="#111827", label="Broad/distributed regime"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=REGIME_COLORS[1], markeredgecolor="#111827", label="Localized/concentrated regime"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#b91c1c", markeredgewidth=1.4, label="Visible damage precedes RKHS transition"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Contact-state evolution and its boundary with visible ice damage", fontsize=17, y=0.995)
    fig.text(
        0.5,
        0.015,
        "Panels summarize descriptive evidence, not a causal pipeline. Thin lines/points are individual specimens; bands are interquartile ranges.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.08, hspace=0.29, wspace=0.24)
    fig.savefig(output, dpi=240)
    plt.close(fig)


def figure_catalog() -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Fig. 1",
            "placement": "main",
            "file": "paper_package/paper_contact_evolution_framework.png",
            "purpose": "Overall evidence framework",
            "uncertainty_representation": "all specimens; median and IQR where trajectories are summarized",
            "allowed_claim": "RKHS describes nonlinear contact-state regimes and their non-universal relation to visible damage.",
            "prohibited_claim": "A causal sequence from loading rate to RKHS transition to fracture has been proven.",
        },
        {
            "figure_id": "Fig. 2",
            "placement": "main",
            "file": "representative_multimodal_comparison.png",
            "purpose": "Synchronized pressure/high-speed examples",
            "uncertainty_representation": "objectively selected rate-group medoids; exact frames and pressure states",
            "allowed_claim": "Pressure fields and side-surface appearance can be inspected at common semantic events.",
            "prohibited_claim": "The displayed medoids represent every specimen or establish event chronology by column order.",
        },
        {
            "figure_id": "Fig. 3",
            "placement": "main",
            "file": "paper_figures/paper_event_pressure_states_iqr.png",
            "purpose": "Event-defined pressure topology",
            "uncertainty_representation": "all specimen trajectories, medians and IQR bands",
            "allowed_claim": "Absolute pressure-topology states differ strongly between rate groups.",
            "prohibited_claim": "The absolute group separation is a causal loading-rate effect.",
        },
        {
            "figure_id": "Fig. 4",
            "placement": "main",
            "file": "paper_figures/paper_incremental_rkhs_iqr.png",
            "purpose": "Initial-state-adjusted nonlinear evolution",
            "uncertainty_representation": "all specimen trajectories, medians and IQR bands",
            "allowed_claim": "Pressure states move nonlinearly away from the initial contact state, with baseline-sensitive group differences.",
            "prohibited_claim": "The trajectory gives a validated fracture probability or universal rate law.",
        },
        {
            "figure_id": "Fig. 5",
            "placement": "main",
            "file": "contact_regime_conditioned/conditional_event_sequence.png",
            "purpose": "RKHS transition and visible-damage chronology",
            "uncertainty_representation": "all 16 specimens shown individually",
            "allowed_claim": "The RKHS transition precedes visible damage in 12/16 specimens but has four documented exceptions.",
            "prohibited_claim": "RKHS transition universally predicts damage onset.",
        },
        {
            "figure_id": "Fig. 6",
            "placement": "main",
            "file": "outlier_mechanism_audit/outlier_mechanism_audit_overview.png",
            "purpose": "Interface-pressure versus visible-damage decoupling and quality limits",
            "uncertainty_representation": "all specimens and explicit quality markers; no aggregate error bars",
            "allowed_claim": "Large contact redistribution and large visible damage need not coincide.",
            "prohibited_claim": "The audit-priority magnitude is a damage score or prediction target.",
        },
        {
            "figure_id": "Fig. S1",
            "placement": "supplement",
            "file": "paper_figures/paper_baseline_adjusted_evolution_iqr.png",
            "purpose": "Within-specimen event-state changes",
            "uncertainty_representation": "raw changes, medians and IQR bands",
            "allowed_claim": "Baseline subtraction removes persistent offsets at the selected semantic states.",
            "prohibited_claim": "IQR bands are confidence intervals.",
        },
        {
            "figure_id": "Fig. S2",
            "placement": "supplement",
            "file": "paper_figures/paper_low_high_effect_matrix.png",
            "purpose": "Effect-size and multiple-testing audit",
            "uncertainty_representation": "Cliff's delta and BH q values; complete intervals retained in CSV/forest plot",
            "allowed_claim": "Absolute low-high state contrasts are large for selected topology variables.",
            "prohibited_claim": "The heatmap replaces reporting complete confidence intervals and exact tests.",
        },
        {
            "figure_id": "Fig. S3",
            "placement": "supplement",
            "file": "paper_figures/paper_spatial_redistribution_iqr.png",
            "purpose": "Event-pair pressure-field redistribution",
            "uncertainty_representation": "all specimens, medians and IQR boxes",
            "allowed_claim": "Pressure load is spatially redistributed between semantic states.",
            "prohibited_claim": "An event-pair arrow establishes temporal or causal direction in every specimen.",
        },
        {
            "figure_id": "Fig. S4",
            "placement": "supplement",
            "file": "early_contact_regimes/early_contact_regime_profiles_iqr.png",
            "purpose": "Rate-blind contact-regime profiles",
            "uncertainty_representation": "all specimen trajectories, medians and IQR bands",
            "allowed_claim": "Two exploratory early-contact state domains are present in the current sample.",
            "prohibited_claim": "The clusters are definitive material failure modes.",
        },
        {
            "figure_id": "Fig. S5",
            "placement": "supplement",
            "file": "contact_regime_conditioned/conditional_rkhs_damage_associations.png",
            "purpose": "Within-regime association audit",
            "uncertainty_representation": "all specimens; permutation p values and BH q values",
            "allowed_claim": "No tested within-regime association survives multiplicity correction.",
            "prohibited_claim": "A non-significant association proves no physical relation exists.",
        },
        {
            "figure_id": "Fig. S6-S14",
            "placement": "supplement",
            "file": "outlier_mechanism_audit/case_plates/*_outlier_case_plate.png",
            "purpose": "Nine specimen-level multimodal audits",
            "uncertainty_representation": "exact specimen states, pressure saturation contours and quality metadata",
            "allowed_claim": "Specific counterexamples, measurement limits and decoupled cases are directly documented.",
            "prohibited_claim": "Selected cases estimate population prevalence.",
        },
    ]
    return pd.DataFrame(rows)


def write_captions(catalog: pd.DataFrame, output: Path) -> None:
    captions = {
        "Fig. 1": (
            "接触状态演化及其与可见冰损伤的边界。a，不使用速率标签获得的早期接触状态几何；b，在局部/集中式接触机制内，中速与高速试件相对于各自40%载荷状态的后期RKHS演化；c，转变—损伤压力拓扑差幅值与最大可见损伤面积之间的解耦；d，RKHS转变与首次持续可见损伤的事件顺序。各面板用于描述证据结构，不表示已经建立的因果链。",
            "Contact-state evolution and its boundary with visible ice damage. (a) Early-contact geometry obtained without rate labels. (b) Later RKHS evolution of medium- and high-rate specimens within the localized/concentrated contact regime, referenced to each specimen's 40%-load state. (c) Decoupling between transition-to-damage pressure-topology change and maximum visible damage area. (d) Specimen-level order of the RKHS transition and first sustained visible damage. The panels organize descriptive evidence and do not constitute a proven causal chain.",
        ),
        "Fig. 2": (
            "客观选取的各速率组代表试件在事件定义状态下的端面压力场和高速侧视图。各列为语义事件对应，不保证按时间顺序排列；真实时序见单试件载荷曲线。压力图下边朝向高速摄像机，未作左右镜像。",
            "End-face pressure fields and high-speed side views for objectively selected rate-group medoids at event-defined states. Columns denote semantic event correspondences and are not necessarily chronological; specimen-level load histories provide the actual order. The lower edge of each pressure map faces the high-speed camera, with no left-right mirroring.",
        ),
        "Fig. 3": (
            "各语义事件状态下的压力拓扑演化。细线表示独立试件，粗线表示组中位数，阴影表示四分位区间。事件列不对全部试件强制统一时间顺序。",
            "Pressure-topology evolution across semantic event states. Thin lines denote independent specimens, thick lines denote group medians, and shaded regions denote interquartile ranges. Event columns do not impose a common chronological order on all specimens.",
        ),
        "Fig. 4": (
            "相对于10%峰值载荷压力状态的RKHS距离演化。显示全部试件轨迹、中位数和四分位区间；该距离刻画接触压力状态偏离初始状态的程度，不是破坏概率。",
            "RKHS-distance evolution relative to the 10%-peak-load pressure state. All specimen trajectories, group medians and interquartile ranges are shown. The distance quantifies departure of the contact-pressure state from its initial state and is not a fracture probability.",
        ),
        "Fig. 5": (
            "两类早期接触机制内RKHS转变与首次持续可见损伤的真实事件顺序。横线仅连接同一试件的两个事件；12/16组中RKHS转变较早，4/16组顺序相反。",
            "Observed order of the RKHS transition and first sustained visible damage within the two early-contact regimes. Horizontal segments connect the two events within each specimen only. The RKHS transition occurs first in 12 of 16 specimens, whereas four specimens show the reverse order.",
        ),
        "Fig. 6": (
            "压力拓扑离群试件的多模态复核优先级及测量质量背景。拓扑差稳健幅值仅用于筛选案例，不表示损伤评分；红色边框标记可见损伤早于RKHS转变的试件。",
            "Multimodal audit priorities and measurement-quality context for pressure-topology outliers. The robust topology-difference magnitude is used only to prioritize case review and is not a damage score. Red outlines identify specimens in which visible damage precedes the RKHS transition.",
        ),
    }
    lines = ["# 论文图件中英文图注", ""]
    for figure_id in catalog.loc[catalog["placement"] == "main", "figure_id"]:
        zh, en = captions[figure_id]
        lines.extend([f"## {figure_id}", "", f"中文：{zh}", "", f"English: {en}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def write_guardrails(output: Path) -> None:
    text = """# 论文结果表达边界

## 可以写入主文的结论

1. RKHS能够以非线性几何形式表征冰—压头端面接触压力状态，并识别接触扩展、局部化和载荷重分配。
2. 当前样本中存在宽域/分布式与局部/集中式两个探索性早期接触状态域；它们与加载速率标签显著相关，但聚类稳定性中等。
3. 低速与高速试件的绝对接触状态明显不同；部分初始状态校正后的多变量轨迹差异在10%和15%基线下仍显著，但对20%和25%基线不稳健。
4. RKHS转变在12/16个试件中早于首次持续可见损伤，但存在4个反序试件，因此只能称为候选接触状态转折，不能称为普遍损伤先兆。
5. 端面压力拓扑变化与侧面可见损伤可以解耦；RKHS更适合作为接触机制描述坐标，而不是独立的破坏或峰值载荷预测器。
6. 所有推断以独立试件为统计单位；主图显示全部试件、中位数与IQR，完整Bootstrap区间和精确置换检验保留在补充材料。

## 不得写入的结论

1. 不得宣称加载速率已经被证明因果性地改变接触机制；速率与批次、端面平整度及摆放可能混杂。
2. 不得宣称RKHS转变能够准确预测裂纹起始、宏观破碎或峰值冰载荷。
3. 不得把高速侧视图中的首次持续可见变化等同于内部首裂时刻。
4. 不得把速率组中位压力热点位置解释为材料固有裂纹源位置。
5. 不得在薄膜饱和、合力比异常或同一压力帧映射两个事件时解释绝对峰值压力或毫秒级状态变化。
6. 不得把IQR称为置信区间，也不得截短或删除完整不确定性区间以改善图面。

## 推荐的论文中心句

本研究提出了一种基于RKHS的多模态冰载荷分析框架，用于刻画单轴压缩过程中端面接触压力状态的非线性演化。结果表明，早期接触状态与加载速率标签相关，但端面压力重分配与侧面可见破坏之间不存在统一时序或一一对应关系。因此，该方法当前最可靠的用途是识别和比较接触机制，而不是进行未经验证的破坏或峰值载荷预测。
"""
    output.write_text(text, encoding="utf-8")


def results_evidence_matrix() -> pd.DataFrame:
    rows = [
        {
            "section": "3.1",
            "topic": "Multimodal dataset and semantic alignment",
            "primary_evidence": "16 independent specimens: low n=3, medium n=5, high n=8",
            "key_statistic": "Event states are specimen-specific and are not forced into one chronology",
            "primary_figure": "Fig. 2",
            "source": "multimodal_event_table.csv; representative_multimodal_comparison.png",
            "claim_strength": "descriptive",
            "required_restriction": "Do not equate first sustained side-view change with internal crack initiation.",
        },
        {
            "section": "3.2",
            "topic": "Nonlinear contact-pressure evolution",
            "primary_evidence": "Specimen trajectories depart nonlinearly from their initial pressure states",
            "key_statistic": "All specimens shown; median and IQR used only as distribution summaries",
            "primary_figure": "Fig. 3; Fig. 4",
            "source": "paper_event_pressure_states_iqr.png; paper_incremental_rkhs_iqr.png",
            "claim_strength": "descriptive-mechanistic",
            "required_restriction": "RKHS distance is not a fracture probability or damage score.",
        },
        {
            "section": "3.3",
            "topic": "Rate-blind early-contact regimes",
            "primary_evidence": "Broad/distributed n=4 and localized/concentrated n=12",
            "key_statistic": "silhouette=0.326; leave-one-out median ARI=0.461; Cramer's V=0.856, p=0.00552",
            "primary_figure": "Fig. 1a; Fig. S4",
            "source": "early_contact_regimes",
            "claim_strength": "exploratory association",
            "required_restriction": "Do not describe the clusters as definitive material failure modes or a causal rate effect.",
        },
        {
            "section": "3.4",
            "topic": "Initial-state-adjusted rate contrasts",
            "primary_evidence": "Low-high incremental trajectories differ at early baselines but not at later baselines",
            "key_statistic": "10%/15% p=0.0061/0.0061; 20%/25% p=0.0848/0.1273",
            "primary_figure": "Fig. 4; Fig. S1-S3",
            "source": "incremental_rkhs_sensitivity; bootstrap_event_analysis",
            "claim_strength": "sensitivity-bounded association",
            "required_restriction": "Report the baseline sensitivity and retain full intervals in the supplement.",
        },
        {
            "section": "3.5",
            "topic": "RKHS transition and visible-damage chronology",
            "primary_evidence": "RKHS transition precedes damage in 12/16 specimens, with four reverse-order cases",
            "key_statistic": "all within-regime association tests q>=0.736",
            "primary_figure": "Fig. 1d; Fig. 5; Fig. S5",
            "source": "contact_regime_conditioned",
            "claim_strength": "counterexample-bounded",
            "required_restriction": "Do not call the RKHS transition a universal damage precursor.",
        },
        {
            "section": "3.6",
            "topic": "Rate comparison within the localized-contact regime",
            "primary_evidence": "Medium n=4 and high n=8 specimens do not separate in the post-40% trajectory",
            "key_statistic": "absolute MMD2=0.1471, p=0.4667; adjusted MMD2=0.1667, p=0.5051",
            "primary_figure": "Fig. 1b",
            "source": "contact_regime_conditioned",
            "claim_strength": "negative exploratory result",
            "required_restriction": "Non-significance does not prove identical physics.",
        },
        {
            "section": "3.7",
            "topic": "Interface-pressure and visible-damage decoupling",
            "primary_evidence": "0.42-1 and 21-4 provide opposite decoupling extremes",
            "key_statistic": "0.42-1: topology rank 1/16 with about 0.4% visible damage; 21-4: about 77.9% damage with near-stable topology",
            "primary_figure": "Fig. 1c; Fig. 6; Fig. S6-S14",
            "source": "outlier_mechanism_audit",
            "claim_strength": "multimodal case evidence",
            "required_restriction": "Audit magnitude is a review priority, not a damage score or population estimate.",
        },
    ]
    return pd.DataFrame(rows)


def write_results_discussion_structure(output: Path) -> None:
    text = """# 论文“结果与讨论”章节结构建议稿

## 总体叙事顺序

全文应从“单试件接触状态如何演化”逐步推进到“试件之间是否存在可重复机制差异”，最后用多模态反例界定RKHS能够解释和不能解释的物理范围。不要以“预测精度”为叙事主线；当前数据最有力的贡献是接触状态表征、机制分区、敏感性检验和压力—损伤解耦证据。

## 3 结果

### 3.1 多模态数据集与事件定义状态

建议用1—2段说明16个独立试件组成低速3组、中速5组和高速8组，`210-7`、`210-9`因当前数据约束不纳入分析。机器载荷、压力薄膜和高速侧视记录按照单试件时间关系对齐，并定义初始接触、RKHS转变、首次持续可见损伤和机器峰值等语义状态。强调语义状态便于跨试件比较，但不同试件的事件顺序并不相同。

对应图件：Fig. 2。压力图下边朝向高速摄像机且不进行左右镜像；侧视可见变化不能等同于内部首裂。

### 3.2 端面接触压力状态的非线性演化

首先展示事件定义状态下接触面积、有效承压面积、压力熵、载荷集中度和压力空间分布的变化，再展示相对于各试件自身10%峰值载荷状态的RKHS距离。结果应写成：不同试件均表现出随加载发展的非线性状态偏移，但轨迹幅度和转折位置具有明显个体差异。RKHS距离是接触压力状态在特征空间中的偏离程度，不是裂纹概率、破坏概率或损伤严重度。

对应图件：Fig. 3、Fig. 4。正文图保留全部试件轨迹、组中位数和IQR；完整Bootstrap区间放入补充材料。

### 3.3 速率盲法识别的早期接触状态域

说明聚类只使用10%—40%峰值载荷区间的9个压力拓扑特征，没有输入速率、峰值载荷或破坏标签。当前样本得到宽域/分布式（n=4）与局部/集中式（n=12）两个探索性状态域。模型平均轮廓系数为0.326，删一试件调整Rand指数中位数为0.461，表明分区存在但稳定性中等。聚类后检验得到状态域与加载速率标签的Cramér's V=0.856、置换p=0.00552。

对应图件：Fig. 1a、Fig. S4。结论只能写成“早期接触状态域与速率标签相关”，不能写成“加载速率已被证明决定接触机制”。

### 3.4 初始状态校正后的速率组差异及敏感性

先报告绝对压力状态的低速—高速分离，再说明减去各试件自身初始状态后的结果。默认10%基线下，低速—高速增量轨迹MMD²=0.4797、exact p=0.0061；0.5—2.0倍核带宽下p=0.0061—0.0121，逐次删除一个低速试件后p=0.0222—0.0444。然而基线改为20%和25%时p升至0.0848和0.1273，表明主要差异形成于非常早期的接触建立阶段，而不是一个贯穿峰前全过程的稳定速率规律。

对应图件：Fig. 3、Fig. 4、Fig. S1—S3。必须同时报告有效结果和失效边界，不得只选择10%基线。

### 3.5 RKHS转变与首次持续可见损伤的事件关系

逐试件事件顺序显示，RKHS转变在12/16个试件中早于首次持续可见损伤，但`0.42-2`、`210-2`、`210-3`和`210-4`呈反序。预先列出的RKHS转变位置、损伤载荷、峰值应力及可见损伤面积关系，在全体和机制条件层面均未通过多重比较校正，全部q值不小于0.736。因此，RKHS转变可称为候选接触状态转折，不能称为普遍损伤先兆。

对应图件：Fig. 1d、Fig. 5、Fig. S5。

### 3.6 相同早期接触机制内的后期速率比较

为降低早期接触机制与速率组近乎共线造成的混杂，只在局部/集中式机制内部比较中速4组和高速8组，并仅使用50%—100%载荷状态。绝对后期轨迹MMD²=0.1471、exact p=0.4667；减去各试件40%状态后MMD²=0.1667、exact p=0.5051，9项单特征轨迹均未通过BH校正。结果说明在相同早期接触状态域内，当前样本没有显示中速与高速后期演化存在可重复分离；这不等同于证明二者物理过程完全相同。

对应图件：Fig. 1b。

### 3.7 端面压力重分配与侧面可见破坏的解耦

以多模态个案结束结果章节。`0.42-1`的压力拓扑变化排序为1/16，而截止前最大可见损伤面积仅约0.4%，应优先解释为端面就位、接触扩展和载荷重分配；`21-4`的最大可见损伤面积约77.9%，而转变与损伤状态间压力拓扑变化接近稳定。这两个相反极端直接表明端面压力场不能完整替代侧面破坏形态。进一步列出薄膜饱和、薄膜/机器合力比异常、压力帧时间错配和同一压力帧映射两个事件等测量边界。

对应图件：Fig. 1c、Fig. 6、Fig. S6—S14。

## 4 讨论

### 4.1 RKHS状态坐标的物理含义

RKHS把接触面积、有效承压面积、熵、载荷集中度和空间分布的协同变化映射为非线性状态距离。它的优势不是给出一个新的“强度值”，而是保留多个接触拓扑指标共同演化时的几何关系，从而识别端面就位、接触扩展、局部化和重分配。

### 4.2 为什么速率关联不能直接解释为速率因果效应

早期状态域与速率标签强相关，但低速组仅3个试件，且速率可能与试验批次、端面平整度、薄膜摆放和初始接触条件混杂。基线敏感性进一步表明主要分离发生在早期接触建立阶段。因此应把结果解释为“与速率标签相关的接触状态差异”，并将端面条件控制试验作为因果验证前提。

### 4.3 压力演化与可见破坏为何可以解耦

端面薄膜观测的是边界载荷传递，侧面高速图像观测的是表观损伤形态；内部裂纹扩展、摩擦约束、局部剥落和三维应力路径并不由任何单一二维观测完整覆盖。压力拓扑显著变化可能只是边界重新就位，而大范围侧面损伤也可能在端面归一化载荷分布近似稳定时发展。

### 4.4 高速形态、分形维数与RKHS的互补关系

分形维数可作为人工确认可见损伤后、且图像质量门控有效区间内的侧面网络复杂度描述量。它不用于补造未观测的细裂纹，也不替代首次持续可见损伤的人工事件定义。讨论中可把压力RKHS视为“边界接触状态坐标”，把损伤面积和质量受控的分形维数视为“侧面表观形态坐标”，二者联合用于描述多模态演化，而不是强行合并为单一损伤分数。

### 4.5 当前方法适合做什么

现阶段可可靠支持：单试件接触状态轨迹描述、早期接触机制分区、压力拓扑转变识别、跨模态事件比较和异常试件溯源。现阶段不能支持：裂纹首发时刻预测、破坏概率预测、峰值载荷外推或普适速率定律。论文创新点应落在“多模态、事件定义、机制条件化且带反例审计的RKHS冰载荷分析框架”。

### 4.6 局限性与后续预测验证设计

主要局限包括样本量小且不平衡、速率与试验条件可能混杂、三源无共同硬件触发、薄膜存在饱和和合力一致性问题、侧视无法观察上下接触断面及内部裂纹。若后续转向预测，需要新增独立批次和更多低速样本，随机化或平衡端面状态和摆放，采用共同触发，并预先固定预测目标、特征窗口、超参数和完全留出试件评价；在此之前不报告“预测准确率”。

## 建议的结果章节收束句

综上，RKHS揭示了冰柱单轴压缩过程中端面接触压力状态的非线性演化及其早期机制差异，但接触压力重分配与侧面可见破坏之间既不存在统一时序，也不存在一一对应关系。该方法当前最可靠的作用是提供一个可审计的接触机制描述坐标，并与高速形态证据互补，而不是单独承担破坏或峰值载荷预测。

## 写作与制图统一规则

1. 统计单位始终是独立试件，不把视频帧或压力帧当作独立样本。
2. 主图显示全部试件、组中位数和IQR；IQR不得称为置信区间。
3. 完整95% Bootstrap区间、Cliff's delta区间、精确置换p值和BH q值保留在补充材料。
4. 对非显著结果使用“当前样本未显示可重复分离”，不写成“证明不存在差异”。
5. 对共现结果使用“同步出现”或“多模态共现”，不写成压力变化导致裂纹。
6. 对`21-1`、`21-3`、`0.42-2`、`210-2`和`210-3`等质量受限个案，必须同时报告相应饱和、合力或同步限制。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data_interim/rkhs_evolution_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_interim/rkhs_evolution_v1/academic/paper_package"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    academic = args.root / "academic"

    assignments = pd.read_csv(academic / "early_contact_regimes" / "early_contact_regime_assignments.csv")
    conditioned_root = academic / "contact_regime_conditioned"
    distances = pd.read_csv(conditioned_root / "localized_post40_rkhs_distances.csv")
    conditioned_mmd = pd.read_csv(conditioned_root / "localized_rate_conditioned_mmd.csv")
    events = pd.read_csv(conditioned_root / "conditional_event_sequence_metrics.csv")
    audit = pd.read_csv(academic / "outlier_mechanism_audit" / "outlier_mechanism_audit_table.csv")

    plot_framework(
        assignments,
        distances,
        conditioned_mmd,
        audit,
        events,
        args.output / "paper_contact_evolution_framework.png",
    )
    catalog = figure_catalog()
    catalog.to_csv(args.output / "paper_figure_catalog.csv", index=False, encoding="utf-8-sig")
    write_captions(catalog, args.output / "paper_figure_captions_zh_en.md")
    write_guardrails(args.output / "paper_results_claims_guardrails_zh.md")
    evidence = results_evidence_matrix()
    evidence.to_csv(args.output / "paper_results_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    write_results_discussion_structure(args.output / "paper_results_discussion_structure_zh.md")
    metadata = {
        "main_figure_count": int((catalog["placement"] == "main").sum()),
        "supplement_catalog_entry_count": int((catalog["placement"] == "supplement").sum()),
        "results_section_count": int(len(evidence)),
        "discussion_section_count": 6,
        "uncertainty_policy": "raw independent specimens plus median and IQR in main figures; full intervals retained in supplement",
        "excluded_specimens": ["-5-rigid-210-7", "-5-rigid-210-9"],
    }
    (args.output / "paper_package_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
