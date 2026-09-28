#!/usr/bin/env python3
"""Run resumable V2 LOSO folds through selection, refinement, fit, and review."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print(subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def passed_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return load_json(path).get("status") == "pass"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/loso/loso_v2_force_based.json"),
    )
    parser.add_argument("--fold-ids", nargs="+", default=["fold_02"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    workspace = config_path.parents[2]
    scripts = workspace / "scripts"
    output_root = workspace / config["output_root"]
    folds = {
        item["fold_id"]: item
        for item in load_json(output_root / "loso_folds.json")["folds"]
    }
    run_root = output_root / "fold_runs_v2_force_based"
    for fold_id in args.fold_ids:
        if fold_id not in folds:
            raise KeyError(f"Unknown fold: {fold_id}")
        started = time.perf_counter()
        fold_root = run_root / fold_id
        selection_root = fold_root / "inner_selection"
        refined_json = selection_root / "inner_selection_refined.json"
        final_root = fold_root / "final"
        final_json = final_root / "fold_result.json"
        print(f"=== {fold_id}: {folds[fold_id]['test_specimen_id']} ===", flush=True)

        if not (args.resume and passed_json(refined_json)):
            run(
                [
                    sys.executable,
                    str(scripts / "tune_loso_fold.py"),
                    "--config",
                    str(config_path),
                    "--fold-id",
                    fold_id,
                    "--output-dir",
                    str(selection_root),
                ]
            )
            run(
                [
                    sys.executable,
                    str(scripts / "refine_loso_inner_selection.py"),
                    "--config",
                    str(config_path),
                    "--fold-id",
                    fold_id,
                    "--selection-root",
                    str(selection_root),
                ]
            )
        else:
            print(f"Reuse passed inner selection: {refined_json}", flush=True)
        if not passed_json(refined_json):
            raise RuntimeError(f"Inner selection did not pass: {refined_json}")

        if not (args.resume and passed_json(final_json)):
            run(
                [
                    sys.executable,
                    str(scripts / "run_loso_fold.py"),
                    "--config",
                    str(config_path),
                    "--fold-id",
                    fold_id,
                    "--selection-json",
                    str(refined_json),
                    "--output-dir",
                    str(final_root),
                ]
            )
        else:
            print(f"Reuse passed outer result: {final_json}", flush=True)
        if not passed_json(final_json):
            raise RuntimeError(f"Outer fold did not pass: {final_json}")

        test_id = folds[fold_id]["test_specimen_id"]
        review_png = final_root / "held_out_probability_review.png"
        run(
            [
                sys.executable,
                str(scripts / "plot_loso_heldout_review.py"),
                "--result-json",
                str(final_json),
                "--predictions-csv",
                str(final_root / "held_out_predictions.csv"),
                "--windows-npz",
                str(
                    output_root
                    / "common_horizon_windows"
                    / test_id
                    / "rkhs_causal_windows.npz"
                ),
                "--output",
                str(review_png),
            ]
        )
        result = load_json(final_json)
        status = {
            "schema_version": "1.0",
            "status": "pass",
            "fold_id": fold_id,
            "held_out_specimen_id": test_id,
            "feature_policy": config["kernel"]["feature_policy"],
            "selection_json": str(refined_json),
            "fold_result_json": str(final_json),
            "review_png": str(review_png),
            "elapsed_seconds_this_invocation": float(time.perf_counter() - started),
            "held_out_metrics": {
                name: values["held_out_metrics"]
                for name, values in result["fits"].items()
            },
        }
        (fold_root / "pipeline_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
