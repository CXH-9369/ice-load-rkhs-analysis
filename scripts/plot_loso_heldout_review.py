#!/usr/bin/env python3
"""Create a focused held-out LOSO probability review figure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import project_dependencies  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"specimen_balanced": "#1f77b4", "rate_balanced": "#ff7f0e"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--windows-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zoom-seconds", type=float, default=180.0)
    args = parser.parse_args()
    result = json.loads(args.result_json.read_text(encoding="utf-8"))
    with args.predictions_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    time_s = np.asarray([float(row["endpoint_time_s"]) for row in rows])
    labels = np.asarray([int(row["label"]) for row in rows])
    tensor_indices = np.asarray(
        [int(row["source_tensor_index_0_based"]) for row in rows], dtype=int
    )
    probability = {
        name: np.asarray([float(row[f"{name}_probability"]) for row in rows])
        for name in COLORS
    }
    thresholds = {
        name: float(result["fits"][name]["model"]["decision_threshold"])
        for name in COLORS
    }
    event_time = float(result["target"]["held_out_event_interval_s"][0])
    with np.load(args.windows_npz, allow_pickle=False) as windows:
        names = windows["mechanics_feature_names"].astype(str).tolist()
        force_index = names.index("machine_force_kn")
        force_kn = windows["mechanics_features"][tensor_indices, -1, force_index]
    order = np.argsort(time_s)
    zoom = time_s >= event_time - args.zoom_seconds

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for name, values in probability.items():
        axes[0].plot(time_s[order], values[order], color=COLORS[name], label=name)
        axes[0].axhline(thresholds[name], color=COLORS[name], linestyle="--", alpha=0.7)
    axes[0].axvline(event_time, color="#c00000", linewidth=1.3, label="event")
    axes[0].set_title("Complete held-out history")
    axes[0].set_xlabel("Window endpoint time (s)")
    axes[0].set_ylabel("Predicted probability")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    for name, values in probability.items():
        axes[1].plot(time_s[zoom], values[zoom], color=COLORS[name], linewidth=1.8, label=name)
        axes[1].axhline(thresholds[name], color=COLORS[name], linestyle="--", alpha=0.7)
    positive = zoom & (labels == 1)
    axes[1].scatter(time_s[positive], labels[positive], s=18, color="#70ad47", label="positive label")
    axes[1].axvline(event_time, color="#c00000", linewidth=1.3, label="event")
    axes[1].set_title(f"Final {args.zoom_seconds:g} s")
    axes[1].set_xlabel("Window endpoint time (s)")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)

    for name, values in probability.items():
        axes[2].plot(force_kn[order], values[order], color=COLORS[name], linewidth=1.4, label=name)
    axes[2].scatter(
        force_kn[labels == 1],
        probability["rate_balanced"][labels == 1],
        s=12,
        color="#70ad47",
        alpha=0.7,
        label="positive windows",
    )
    axes[2].set_title("Probability versus applied force")
    axes[2].set_xlabel("Machine force (kN)")
    axes[2].set_ylabel("Predicted probability")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"{result['held_out_specimen_id']} — V2 force-based held-out review",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
