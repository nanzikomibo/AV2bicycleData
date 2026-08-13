from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import DEFAULT_ARGO_FILE, DEFAULT_OUTPUT_DIR
from utils.io_utils import ensure_dir, save_csv
from utils.metrics import canonicalize_av2, pair_duration_table, pair_level_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AV2 features using explicit string actor types and source lateral offset.")
    parser.add_argument("--argo-file", type=Path, default=DEFAULT_ARGO_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.argo_file.exists():
        raise FileNotFoundError(args.argo_file)
    processed_dir = ensure_dir(args.output_dir / "processed")
    tables_dir = ensure_dir(args.output_dir / "tables" / "preprocessing")

    print(f"[INFO] Reading AV2 data: {args.argo_file}", flush=True)
    raw = pd.read_csv(args.argo_file, low_memory=False)
    features, audit = canonicalize_av2(raw)

    # These assertions specifically prevent the historical bus/bicycle class-code error.
    bicycle_raw = features["leader_type_raw"].astype(str).str.lower().eq("bicycle")
    bus_raw = features["leader_type_raw"].astype(str).str.lower().eq("bus")
    if not features.loc[bicycle_raw, "leader_group"].eq("bicycle").all():
        raise RuntimeError("Classification validation failed: raw bicycle leaders were not retained as bicycle.")
    if features.loc[bus_raw, "leader_group"].eq("bicycle").any():
        raise RuntimeError("Classification validation failed: bus leaders entered the bicycle group.")
    if features["lateral_offset"].isna().any():
        raise RuntimeError("AV2 d_perp/lateral_offset unexpectedly contains missing values.")

    save_csv(features, processed_dir / "argo_features.csv", float_format="%.10g")
    save_csv(audit, tables_dir / "argo_raw_classification_audit.csv")
    duration = pair_duration_table(features)
    save_csv(duration, tables_dir / "argo_pair_duration.csv", float_format="%.10g")
    pair_table = pair_level_metrics(features)
    save_csv(pair_table, processed_dir / "argo_pair_level_unfiltered.csv", float_format="%.10g")

    raw_pairs = raw.groupby("pair_type", dropna=False).agg(frames=("pair_segment_id", "size"), pairs=("pair_segment_id", "nunique")).reset_index()
    canonical_pairs = features.groupby("analysis_group", dropna=False).agg(frames=("pair_id", "size"), pairs=("pair_id", "nunique")).reset_index()
    save_csv(raw_pairs, tables_dir / "argo_pair_counts_by_raw_type.csv")
    save_csv(canonical_pairs, tables_dir / "argo_pair_counts_by_analysis_group.csv")
    overview = pd.DataFrame([{
        "source_file": str(args.argo_file), "raw_rows": len(raw), "processed_rows": len(features),
        "pairs": features["pair_id"].nunique(), "scenarios": features["scenario_id"].nunique(),
        "raw_bicycle_av_pairs": raw.loc[raw["pair_type"].eq("bicycle-AV"), "pair_segment_id"].nunique(),
        "raw_bicycle_hv_pairs": raw.loc[raw["pair_type"].eq("bicycle-HV"), "pair_segment_id"].nunique(),
        "lateral_offset_missing_ratio": features["lateral_offset"].isna().mean(),
    }])
    save_csv(overview, tables_dir / "argo_input_overview.csv")
    print(f"[DONE] AV2: {len(features):,} frames, {features['pair_id'].nunique():,} pairs", flush=True)


if __name__ == "__main__":
    main()

