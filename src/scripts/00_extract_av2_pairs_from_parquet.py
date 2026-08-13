"""Download official AV2 scenario Parquet files and extract bicycle-leading pairs.

The source data are not distributed with this repository. AV2 motion-forecasting
training scenarios are publicly available at:
    s3://argoverse/datasets/av2/motion-forecasting/train/

Install the AWS CLI, then either download scenarios yourself or let this script
download them one at a time without credentials:

    aws s3 cp --no-sign-request \
      s3://argoverse/datasets/av2/motion-forecasting/train/<SCENE_ID>/scenario_<SCENE_ID>.parquet \
      data/raw/scenario_parquet/scenario_<SCENE_ID>.parquet

Examples:
    python src/scripts/00_extract_av2_pairs_from_parquet.py \
      --parquet-dir data/raw/scenario_parquet --extract-only

    python src/scripts/00_extract_av2_pairs_from_parquet.py \
      --parquet-dir data/raw/scenario_parquet --download --start 1 --stop 100

The script treats a vehicle whose raw ``track_id`` is exactly ``AV`` as an
autonomous vehicle and all other ``vehicle`` tracks as human-driven vehicles.
It writes a frame-level CSV compatible with ``01_prepare_argo_features.py``.
No Parquet files are downloaded or processed unless this script is explicitly run.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


S3_TRAIN_ROOT = "s3://argoverse/datasets/av2/motion-forecasting/train"
DT_S = 0.1
MAX_SPACING_M = 30.0
MAX_LATERAL_OFFSET_M = 2.0
MIN_FOLLOWER_SPEED_MPS = 1.0
MIN_BICYCLE_SPEED_MPS = 2.0
MIN_TIME_HEADWAY_S = 0.5
MAX_TIME_HEADWAY_S = 10.0
MIN_HEADING_COS = math.cos(math.radians(30.0))
MIN_SEGMENT_FRAMES = 50


def parse_args() -> argparse.Namespace:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Extract AV/HV-following-bicycle interactions from official AV2 scenario Parquet files."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=repository_root / "data" / "raw" / "scenario_parquet",
        help="Directory containing downloaded scenario_*.parquet files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=repository_root / "outputs" / "from_official_parquet" / "av2_new_leader_follower_frame_level_dataset.csv",
        help="Frame-level leader-follower CSV to create.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=repository_root / "outputs" / "from_official_parquet" / "av2_new_leader_follower_segment_summary.csv",
        help="Segment-level audit CSV to create.",
    )
    parser.add_argument("--download", action="store_true", help="Download official Parquet files before extraction.")
    parser.add_argument("--extract-only", action="store_true", help="Process only Parquet files already present locally.")
    parser.add_argument("--start", type=int, default=1, help="First S3 scene index when using --download (1-based).")
    parser.add_argument("--stop", type=int, default=None, help="Last S3 scene index when using --download (inclusive).")
    parser.add_argument("--aws", default="aws", help="AWS CLI executable.")
    return parser.parse_args()


def natural_key(value: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value.name)]


def list_s3_scene_ids(aws_executable: str) -> list[str]:
    command = [aws_executable, "s3", "ls", f"{S3_TRAIN_ROOT}/", "--no-sign-request"]
    listing = subprocess.check_output(command, text=True)
    return [line.split()[1].rstrip("/") for line in listing.splitlines() if line.split()[:1] == ["PRE"]]


def download_scene(scene_id: str, parquet_dir: Path, aws_executable: str) -> Path:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    destination = parquet_dir / f"scenario_{scene_id}.parquet"
    if not destination.exists():
        source = f"{S3_TRAIN_ROOT}/{scene_id}/scenario_{scene_id}.parquet"
        subprocess.check_call([aws_executable, "s3", "cp", source, str(destination), "--no-sign-request"])
    return destination


def normalize_object_type(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"cyclist", "bicyclist", "bicycle", "riderless_bicycle"}:
        return "bicycle"
    if text in {"vehicle", "car"}:
        return "vehicle"
    return text


def read_scenario(path: Path) -> pd.DataFrame:
    required = {
        "track_id", "object_type", "timestep", "position_x", "position_y", "heading", "velocity_x", "velocity_y"
    }
    data = pd.read_parquet(path)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    if "scenario_id" not in data.columns:
        data["scenario_id"] = path.stem.removeprefix("scenario_")
    if "observed" not in data.columns:
        data["observed"] = True
    data = data.copy()
    data["track_id"] = data["track_id"].astype(str)
    data["scenario_id"] = data["scenario_id"].astype(str)
    data["actor_type"] = data["object_type"].map(normalize_object_type)
    data = data[data["actor_type"].isin(["bicycle", "vehicle"])].copy()
    data = data.sort_values(["scenario_id", "track_id", "timestep"]).reset_index(drop=True)
    data["speed"] = np.hypot(data["velocity_x"], data["velocity_y"])
    grouped = data.groupby(["scenario_id", "track_id"], sort=False)
    data["acceleration"] = (
        data["velocity_x"] * grouped["velocity_x"].diff().div(DT_S)
        + data["velocity_y"] * grouped["velocity_y"].diff().div(DT_S)
    ).div(data["speed"].replace(0, np.nan))
    data["acceleration"] = grouped["acceleration"].transform(lambda value: value.bfill().ffill()).fillna(0.0)
    return data


def extract_frame_candidates(scene: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestep, frame in scene.groupby("timestep", sort=True):
        bicycles = frame[frame["actor_type"].eq("bicycle")]
        vehicles = frame[frame["actor_type"].eq("vehicle")]
        if bicycles.empty or vehicles.empty:
            continue
        bike_x = bicycles["position_x"].to_numpy(float)
        bike_y = bicycles["position_y"].to_numpy(float)
        bike_vx = bicycles["velocity_x"].to_numpy(float)
        bike_vy = bicycles["velocity_y"].to_numpy(float)
        bike_speed = bicycles["speed"].to_numpy(float)
        bike_heading = bicycles["heading"].to_numpy(float)
        bike_acc = bicycles["acceleration"].to_numpy(float)
        bike_id = bicycles["track_id"].to_numpy(str)
        bike_observed = bicycles["observed"].astype(bool).to_numpy()
        scene_id = str(frame["scenario_id"].iloc[0])

        for vehicle in vehicles.itertuples(index=False):
            follower_speed = float(vehicle.speed)
            if follower_speed < MIN_FOLLOWER_SPEED_MPS or not np.isfinite(vehicle.heading):
                continue
            cosine, sine = math.cos(float(vehicle.heading)), math.sin(float(vehicle.heading))
            dx = bike_x - float(vehicle.position_x)
            dy = bike_y - float(vehicle.position_y)
            d_parallel = dx * cosine + dy * sine
            d_perp = -dx * sine + dy * cosine
            heading_cos = (
                float(vehicle.velocity_x) * bike_vx + float(vehicle.velocity_y) * bike_vy
            ) / (follower_speed * bike_speed + 1e-9)
            headway = d_parallel / follower_speed
            valid = (
                (d_parallel > 0)
                & (d_parallel <= MAX_SPACING_M)
                & (np.abs(d_perp) <= MAX_LATERAL_OFFSET_M)
                & (bike_speed >= MIN_BICYCLE_SPEED_MPS)
                & (heading_cos >= MIN_HEADING_COS)
                & (headway >= MIN_TIME_HEADWAY_S)
                & (headway <= MAX_TIME_HEADWAY_S)
            )
            candidates = np.where(valid)[0]
            if candidates.size == 0:
                continue
            leader = candidates[np.argmin(d_parallel[candidates])]
            follower_type = "AV" if str(vehicle.track_id).upper() == "AV" else "HV"
            rows.append(
                {
                    "scene_id": scene_id,
                    "local_time_s": float(timestep) * DT_S,
                    "timestep": int(timestep),
                    "pair_type": f"bicycle-{follower_type}",
                    "leader_type": "bicycle",
                    "follower_type": follower_type,
                    "leader_track_id": bike_id[leader],
                    "follower_track_id": str(vehicle.track_id),
                    "l_Local_X": bike_x[leader],
                    "l_Local_Y": bike_y[leader],
                    "f_Local_X": float(vehicle.position_x),
                    "f_Local_Y": float(vehicle.position_y),
                    "l_Vehicle_Length": 1.8,
                    "l_Vehicle_Width": 0.6,
                    "f_Vehicle_Length": 4.5,
                    "f_Vehicle_Width": 1.8,
                    "l_Vehicle_Class": 4,
                    "f_Vehicle_Class": 1,
                    "l_Speed": bike_speed[leader],
                    "f_Speed": follower_speed,
                    "l_Acc": bike_acc[leader],
                    "f_Acc": float(vehicle.acceleration),
                    "d_parallel": d_parallel[leader],
                    "d_perp": d_perp[leader],
                    "Spacing": d_parallel[leader] - (1.8 + 4.5) / 2,
                    "time_headway": headway[leader],
                    "Speed_difference_follower_minus_leader": follower_speed - bike_speed[leader],
                    "leader_av_flag": 0,
                    "follower_av_flag": int(follower_type == "AV"),
                    "l_observed": bool(bike_observed[leader]),
                    "f_observed": bool(vehicle.observed),
                    "heading_cos": heading_cos[leader],
                }
            )
    return pd.DataFrame(rows)


def split_continuous_segments(frame_pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame_pairs.empty:
        return frame_pairs, pd.DataFrame()
    pairs = frame_pairs.sort_values(["scene_id", "leader_track_id", "follower_track_id", "timestep"]).copy()
    keys = ["scene_id", "leader_track_id", "follower_track_id", "pair_type"]
    gap = pairs.groupby(keys, sort=False)["timestep"].diff()
    pairs["segment_index"] = (gap.isna() | gap.gt(1)).groupby([pairs[key] for key in keys], sort=False).cumsum()
    pairs["pair_segment_id"] = (
        pairs["scene_id"]
        + "_L"
        + pairs["leader_track_id"].astype(str)
        + "_F"
        + pairs["follower_track_id"].astype(str)
        + "_S"
        + pairs["segment_index"].astype(str)
    )
    counts = pairs.groupby("pair_segment_id", sort=False)["timestep"].transform("size")
    pairs = pairs[counts >= MIN_SEGMENT_FRAMES].copy()
    pairs["pair_id"] = pairs["pair_segment_id"]
    summary = (
        pairs.groupby("pair_segment_id", dropna=False)
        .agg(
            scene_id=("scene_id", "first"),
            pair_type=("pair_type", "first"),
            leader_type=("leader_type", "first"),
            follower_type=("follower_type", "first"),
            leader_track_id=("leader_track_id", "first"),
            follower_track_id=("follower_track_id", "first"),
            start_timestep=("timestep", "min"),
            end_timestep=("timestep", "max"),
            n_frames=("timestep", "size"),
            mean_spacing=("Spacing", "mean"),
            mean_time_headway=("time_headway", "mean"),
        )
        .reset_index()
    )
    summary["duration_s"] = summary["n_frames"] * DT_S
    return pairs.drop(columns="segment_index"), summary


def main() -> None:
    args = parse_args()
    if args.download and args.extract_only:
        raise ValueError("Use either --download or --extract-only, not both.")
    if args.download:
        scene_ids = list_s3_scene_ids(args.aws)
        stop = args.stop or len(scene_ids)
        for scene_id in scene_ids[args.start - 1 : stop]:
            download_scene(scene_id, args.parquet_dir, args.aws)
    parquet_files = sorted(args.parquet_dir.glob("scenario_*.parquet"), key=natural_key)
    if not parquet_files:
        raise FileNotFoundError(f"No scenario_*.parquet files found in {args.parquet_dir}")
    frame_parts: list[pd.DataFrame] = []
    for path in parquet_files:
        candidates = extract_frame_candidates(read_scenario(path))
        if not candidates.empty:
            frame_parts.append(candidates)
    final_frames, summary = split_continuous_segments(
        pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    final_frames.to_csv(args.output_file, index=False, encoding="utf-8-sig")
    summary.to_csv(args.summary_file, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(final_frames):,} frame records to {args.output_file}")
    print(f"Wrote {len(summary):,} interaction segments to {args.summary_file}")


if __name__ == "__main__":
    main()
