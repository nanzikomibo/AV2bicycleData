from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import BENCHMARK_DT_S, DEFAULT_OLD_DIR, DEFAULT_OUTPUT_DIR, OLD_FILES
from utils.io_utils import ensure_dir, require_files, save_csv
from utils.metrics import benchmark_usecols, canonicalize_benchmark, pair_level_metrics


CORE_COLUMNS = [
    "dataset_source", "dataset_name", "scenario_id", "frame", "time_s", "pair_id", "follower_id", "leader_id",
    "raw_pair_type", "follower_type_raw", "leader_type_raw", "follower_group", "leader_group", "analysis_group",
    "follower_is_av", "follower_is_hv", "leader_is_bicycle", "f_v", "l_v", "f_a_source", "l_a_source",
    "f_a_from_speed", "l_a_from_speed", "f_a_used", "l_a_used", "f_jerk", "l_jerk",
    "spacing", "d_parallel", "lateral_offset", "abs_lateral_offset", "relative_speed_f_minus_l", "relative_acceleration_f_minus_l",
    "time_headway", "ttc", "mttc", "drac", "tit_deficit_3p0", "tit_deficit_1p5", "interaction_state", "motion_state", "duration_s", "dt", "f_speed_from_pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess seven naturalistic benchmark datasets with strict column mapping.")
    parser.add_argument("--old-dir", type=Path, default=DEFAULT_OLD_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dt", type=float, default=BENCHMARK_DT_S)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = require_files(args.old_dir, OLD_FILES)
    processed_dir = ensure_dir(args.output_dir / "processed" / "benchmark")
    tables_dir = ensure_dir(args.output_dir / "tables" / "preprocessing")

    overview_rows = []
    class_rows = []
    acceleration_rows = []
    for path in files:
        dataset_name = path.stem.replace("combined_", "").replace("_leader_follower", "")
        usecols = benchmark_usecols(path)
        print(f"[INFO] Reading benchmark {dataset_name}: {path.name} ({len(usecols)} selected columns)", flush=True)
        raw = pd.read_csv(path, usecols=usecols, low_memory=False)
        raw_rows = len(raw)
        features = canonicalize_benchmark(raw, dataset_name, dt_s=args.dt)
        del raw
        gc.collect()

        valid_geometry = features[["l_x", "l_y", "f_x", "f_y"]].notna().all(axis=1)
        features = features.loc[valid_geometry].copy()
        keep = [c for c in CORE_COLUMNS if c in features.columns]
        output_path = processed_dir / f"{dataset_name}_features.csv"
        save_csv(features[keep], output_path, float_format="%.10g")

        pairs = pair_level_metrics(features)
        save_csv(pairs, processed_dir / f"{dataset_name}_pair_level_unfiltered.csv", float_format="%.10g")
        class_table = features.groupby(["follower_type_raw", "leader_type_raw", "leader_group"], dropna=False).agg(
            frames=("pair_id", "size"), pairs=("pair_id", "nunique")
        ).reset_index()
        class_table.insert(0, "dataset_name", dataset_name)
        class_rows.append(class_table)

        acc = pd.to_numeric(features["f_a_used"], errors="coerce")
        jerk = pd.to_numeric(features["f_jerk"], errors="coerce")
        acceleration_rows.append({
            "dataset_name": dataset_name, "frames": len(features), "pairs": features["pair_id"].nunique(),
            "f_a_used_mean": acc.mean(), "f_a_used_median": acc.median(), "f_a_used_std": acc.std(),
            "f_a_used_p01": acc.quantile(0.01), "f_a_used_p99": acc.quantile(0.99),
            "f_jerk_mean": jerk.mean(), "f_jerk_median": jerk.median(), "f_jerk_std": jerk.std(),
            "f_jerk_p01": jerk.quantile(0.01), "f_jerk_p99": jerk.quantile(0.99),
        })
        overview_rows.append({
            "dataset_name": dataset_name, "source_file": str(path), "raw_rows": raw_rows,
            "valid_geometry_rows": len(features), "pairs": features["pair_id"].nunique(),
            "median_dt_s": features["dt"].median(), "median_duration_s": pairs["duration_s"].median(),
            "processed_file": str(output_path),
        })
        print(f"[DONE] {dataset_name}: {len(features):,} frames, {features['pair_id'].nunique():,} pairs", flush=True)
        del features, pairs
        gc.collect()

    overview = pd.DataFrame(overview_rows)
    acceleration = pd.DataFrame(acceleration_rows)
    save_csv(overview, tables_dir / "benchmark_input_overview.csv")
    save_csv(pd.concat(class_rows, ignore_index=True), tables_dir / "benchmark_classification_audit.csv")
    save_csv(acceleration, tables_dir / "benchmark_acceleration_jerk_validation.csv")

    if not acceleration.empty:
        weighted_acc = (acceleration["f_a_used_mean"] * acceleration["frames"]).sum() / acceleration["frames"].sum()
        if not abs(weighted_acc) < 10:
            raise RuntimeError(f"Benchmark acceleration validation failed: weighted mean={weighted_acc}")
        # Six road-trajectory files are 25 Hz. TUM carries an explicit 12.5 Hz
        # timestamp and must retain its source 0.08 s interval.
        expected_dt = overview["dataset_name"].map({"TUM": 0.08}).fillna(args.dt)
        if not (overview["median_dt_s"].sub(expected_dt).abs() < 0.005).all():
            raise RuntimeError("Benchmark timestep validation failed (expected 0.04 s, or source 0.08 s for TUM).")


if __name__ == "__main__":
    main()
