from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-9


def compute_mttc(
    spacing: pd.Series,
    relative_speed: pd.Series,
    relative_acceleration: pd.Series,
) -> pd.Series:
    """Return the smallest positive constant-acceleration collision time.

    The kinematic equation is ``0.5 * da * t**2 + dv * t - spacing = 0``.
    Frames without a positive real collision time are represented by infinity.
    """
    gap = pd.to_numeric(spacing, errors="coerce")
    dv = pd.to_numeric(relative_speed, errors="coerce")
    da = pd.to_numeric(relative_acceleration, errors="coerce")
    result = pd.Series(np.inf, index=gap.index, dtype="float64")
    result.loc[gap <= 0] = 0.0

    linear = gap.gt(0) & da.abs().lt(EPS) & dv.gt(0)
    result.loc[linear] = gap.loc[linear] / dv.loc[linear]

    quadratic = gap.gt(0) & da.abs().ge(EPS)
    discriminant = dv.pow(2) + 2.0 * da * gap
    valid = quadratic & discriminant.ge(0)
    if valid.any():
        root = np.sqrt(discriminant.loc[valid])
        denominator = da.loc[valid]
        first = (-dv.loc[valid] - root) / denominator
        second = (-dv.loc[valid] + root) / denominator
        roots = pd.concat([first.where(first.gt(0)), second.where(second.gt(0))], axis=1)
        result.loc[valid] = roots.min(axis=1).fillna(np.inf)
    return result


def numeric(df: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce")


def text(df: pd.DataFrame, name: str, default="unknown") -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[name].astype("string").fillna(default).str.strip()


def normalize_av2_actor_type(values: pd.Series) -> pd.Series:
    """Normalize the reliable AV2 string type; never infer from numeric class codes."""
    x = values.astype("string").fillna("unknown").str.strip().str.lower()
    out = pd.Series("unknown", index=x.index, dtype="object")
    out[x.isin(["av", "autonomous", "autonomous vehicle"])] = "AV"
    out[x.isin(["hv", "vehicle", "car", "human vehicle", "human-driven vehicle"])] = "HV"
    out[x.str.contains("bicycle|cyclist|bike", regex=True, na=False)] = "bicycle"
    out[x.str.contains("bus", regex=True, na=False)] = "bus"
    out[x.str.contains("motorcycle|motorbike", regex=True, na=False)] = "motorcycle"
    out[x.str.contains("pedestrian|person", regex=True, na=False)] = "pedestrian"
    return out


def normalize_benchmark_type(values: pd.Series) -> pd.Series:
    x = values.astype("string").fillna("unknown").str.strip().str.lower()
    out = pd.Series("unknown", index=x.index, dtype="object")
    out[x.str.contains("bicycle|cyclist|bike", regex=True, na=False)] = "bicycle"
    out[x.str.contains("pedestrian|person", regex=True, na=False)] = "pedestrian"
    out[x.str.contains("motorcycle|motorbike", regex=True, na=False)] = "motorcycle"
    out[x.str.contains("bus", regex=True, na=False)] = "bus"
    out[x.str.contains("car|truck|van|vehicle", regex=True, na=False)] = "vehicle"
    return out


def av2_leader_group(values: pd.Series) -> pd.Series:
    actor = normalize_av2_actor_type(values)
    return actor.replace({"AV": "vehicle", "HV": "vehicle"})


def _safe_dt(out: pd.DataFrame) -> pd.Series:
    dt = out.groupby("pair_id", sort=False)["time_s"].diff()
    positive = dt[(dt > 0) & np.isfinite(dt)]
    fallback = positive.median()
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.1
    return dt.where((dt > 0) & np.isfinite(dt), fallback)


def _fill_dimensions(length: pd.Series, width: pd.Series, actor_type: pd.Series) -> tuple[pd.Series, pd.Series]:
    defaults_length = actor_type.map({"bicycle": 1.8, "motorcycle": 2.2, "bus": 12.0, "vehicle": 4.5}).fillna(4.5)
    defaults_width = actor_type.map({"bicycle": 0.7, "motorcycle": 0.8, "bus": 2.5, "vehicle": 1.8}).fillna(1.8)
    length = pd.to_numeric(length, errors="coerce")
    width = pd.to_numeric(width, errors="coerce")
    length = length.where((length > 0.2) & (length < 30), defaults_length)
    width = width.where((width > 0.2) & (width < 5), defaults_width)
    return length, width


def add_common_metrics(df: pd.DataFrame, preserve_acceleration: bool = True) -> pd.DataFrame:
    """Compute the single authoritative metric set used by all six experiments."""
    out = df.sort_values(["pair_id", "time_s", "frame"], kind="mergesort").reset_index(drop=True).copy()
    out["dt"] = _safe_dt(out)
    group = out.groupby("pair_id", sort=False)

    for prefix in ("f", "l"):
        out[f"{prefix}_speed_from_pos"] = np.sqrt(
            group[f"{prefix}_x"].diff() ** 2 + group[f"{prefix}_y"].diff() ** 2
        ) / out["dt"]
        acc_from_speed = group[f"{prefix}_v"].diff() / out["dt"]
        out[f"{prefix}_a_from_speed"] = acc_from_speed
        source = pd.to_numeric(out.get(f"{prefix}_a_source"), errors="coerce")
        finite = source.replace([np.inf, -np.inf], np.nan).dropna()
        source_is_plausible = bool(
            preserve_acceleration
            and len(finite) > 0
            and finite.abs().median() < 5.0
            and finite.abs().quantile(0.995) < 20.0
        )
        out[f"{prefix}_a_used"] = source.where(source.notna(), acc_from_speed) if source_is_plausible else acc_from_speed

    out["f_jerk"] = out.groupby("pair_id", sort=False)["f_a_used"].diff() / out["dt"]
    out["l_jerk"] = out.groupby("pair_id", sort=False)["l_a_used"].diff() / out["dt"]
    out["relative_speed_f_minus_l"] = out["f_v"] - out["l_v"]
    out["relative_acceleration_f_minus_l"] = out["f_a_used"] - out["l_a_used"]
    out["time_headway"] = out["spacing"] / out["f_v"].where(out["f_v"] > 0.1, np.nan)
    closing = out["relative_speed_f_minus_l"] > 0.1
    out["ttc"] = np.inf
    out.loc[closing, "ttc"] = out.loc[closing, "spacing"] / out.loc[closing, "relative_speed_f_minus_l"]
    out.loc[out["spacing"] <= 0, "ttc"] = 0.0
    out["mttc"] = compute_mttc(
        out["spacing"], out["relative_speed_f_minus_l"], out["relative_acceleration_f_minus_l"]
    )
    out["drac"] = 0.0
    out.loc[closing & (out["spacing"] > 0), "drac"] = (
        out.loc[closing & (out["spacing"] > 0), "relative_speed_f_minus_l"].pow(2)
        / (2.0 * out.loc[closing & (out["spacing"] > 0), "spacing"])
    )
    for threshold in (3.0, 1.5):
        label = str(threshold).replace(".", "p")
        out[f"tit_deficit_{label}"] = (threshold - out["mttc"]).clip(lower=0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["abs_lateral_offset"] = out["lateral_offset"].abs()
    out["interaction_state"] = "following"
    out.loc[out["f_v"] < 0.5, "interaction_state"] = "near_stationary"
    out.loc[(out["f_v"] >= 0.5) & (out["relative_speed_f_minus_l"] > 0.5), "interaction_state"] = "approaching"
    out.loc[(out["f_v"] >= 0.5) & (out["relative_speed_f_minus_l"] < -0.5), "interaction_state"] = "receding"
    out["motion_state"] = "steady"
    out.loc[out["f_a_used"] > 0.5, "motion_state"] = "accelerating"
    out.loc[out["f_a_used"] < -0.5, "motion_state"] = "decelerating"

    duration = pair_duration_table(out)[["pair_id", "duration_s"]]
    out = out.merge(duration, on="pair_id", how="left", validate="many_to_one")
    return out


def canonicalize_av2(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "scene_id", "local_time_s", "pair_segment_id", "pair_type", "leader_type", "follower_type",
        "leader_track_id", "follower_track_id", "l_Local_X", "l_Local_Y", "f_Local_X", "f_Local_Y",
        "l_Vehicle_Length", "l_Vehicle_Width", "f_Vehicle_Length", "f_Vehicle_Width",
        "l_Speed", "f_Speed", "l_Acc", "f_Acc", "d_parallel", "d_perp", "Spacing",
        "leader_av_flag", "follower_av_flag",
    ]
    missing = [name for name in required if name not in raw.columns]
    if missing:
        raise ValueError(f"AV2 input is missing required columns: {missing}")

    out = pd.DataFrame(index=raw.index)
    out["dataset_source"] = "argoai"
    out["dataset_name"] = "ArgoAI"
    out["scenario_id"] = text(raw, "scene_id")
    out["frame"] = numeric(raw, "timestep")
    out["time_s"] = numeric(raw, "local_time_s")
    out["pair_id"] = text(raw, "pair_segment_id")
    out["follower_id"] = text(raw, "follower_track_id")
    out["leader_id"] = text(raw, "leader_track_id")
    out["raw_pair_type"] = text(raw, "pair_type")
    out["follower_type_raw"] = text(raw, "follower_type")
    out["leader_type_raw"] = text(raw, "leader_type")
    out["follower_group"] = normalize_av2_actor_type(out["follower_type_raw"])
    out["leader_group"] = av2_leader_group(out["leader_type_raw"])
    out["analysis_group"] = out["follower_group"] + "-" + out["leader_group"]
    out["follower_is_av"] = out["follower_group"].eq("AV")
    out["follower_is_hv"] = out["follower_group"].eq("HV")
    out["leader_is_bicycle"] = out["leader_group"].eq("bicycle")

    mapping = {
        "f_x": "f_Local_X", "f_y": "f_Local_Y", "l_x": "l_Local_X", "l_y": "l_Local_Y",
        "f_length": "f_Vehicle_Length", "f_width": "f_Vehicle_Width",
        "l_length": "l_Vehicle_Length", "l_width": "l_Vehicle_Width",
        "f_v": "f_Speed", "l_v": "l_Speed", "f_a_source": "f_Acc", "l_a_source": "l_Acc",
    }
    for target, source in mapping.items():
        out[target] = numeric(raw, source)
    out["spacing"] = numeric(raw, "Spacing")
    out["d_parallel"] = numeric(raw, "d_parallel")
    out["lateral_offset"] = numeric(raw, "d_perp")
    out["source_time_headway"] = numeric(raw, "time_headway")
    out["source_relative_speed"] = numeric(raw, "Speed_difference_follower_minus_leader")

    follower_dim_type = out["follower_group"].replace({"AV": "vehicle", "HV": "vehicle"})
    out["f_length"], out["f_width"] = _fill_dimensions(out["f_length"], out["f_width"], follower_dim_type)
    out["l_length"], out["l_width"] = _fill_dimensions(out["l_length"], out["l_width"], out["leader_group"])
    out = add_common_metrics(out, preserve_acceleration=True)

    audit = raw.groupby(["leader_type", "l_Vehicle_Class", "follower_type", "pair_type"], dropna=False).agg(
        frames=("pair_segment_id", "size"), pairs=("pair_segment_id", "nunique")
    ).reset_index()
    return out, audit


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def _series_by_candidates(raw: pd.DataFrame, candidates: Iterable[str], default=np.nan, is_text=False) -> pd.Series:
    col = _first_existing(raw.columns, candidates)
    if is_text:
        return text(raw, col, str(default)) if col else pd.Series(str(default), index=raw.index, dtype="object")
    return numeric(raw, col, default) if col else pd.Series(default, index=raw.index, dtype="float64")


def benchmark_usecols(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    candidate_sets = [
        ["recordingId", "recording_id"], ["trackId", "track_id"], ["frame", "timestamp"], ["trackLifetime"],
        ["xCenter", "translation_x"], ["yCenter", "translation_y"], ["heading", "heading_deg"],
        ["width", "dimension_y"], ["length", "dimension_x"], ["xVelocity", "velocity_x"],
        ["yVelocity", "velocity_y"], ["lonVelocity", "v_s"], ["lonAcceleration"],
        ["xAcceleration", "acceleration_x"], ["yAcceleration", "acceleration_y"], ["class", "category", "subject_category"],
        ["leaderId"], ["leader_id"], ["leader_track_id"], ["leaderType", "leader_category"],
        ["leader_xCenter", "leader_translation_x"], ["leader_yCenter", "leader_translation_y"],
        ["leader_heading", "leader_heading_deg"], ["leader_xVelocity", "leader_velocity_x"],
        ["leader_yVelocity", "leader_velocity_y"], ["leader_lonVelocity", "leader_v_s"],
        ["leader_lonAcceleration"], ["leader_xAcceleration", "leader_acceleration_x"],
        ["leader_yAcceleration", "leader_acceleration_y"], ["leader_dimension_x"], ["leader_dimension_y"],
    ]
    selected = []
    for candidates in candidate_sets:
        col = _first_existing(header, candidates)
        if col and col not in selected:
            selected.append(col)
    return selected


def canonicalize_benchmark(raw: pd.DataFrame, dataset_name: str, dt_s: float = 0.04) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["dataset_source"] = "benchmark"
    out["dataset_name"] = dataset_name
    recording = _series_by_candidates(raw, ["recordingId", "recording_id"], dataset_name, is_text=True)
    out["scenario_id"] = recording
    out["frame"] = _series_by_candidates(raw, ["frame"])
    explicit_time = _series_by_candidates(raw, ["timestamp"])
    if explicit_time.notna().any():
        out["time_s"] = explicit_time
        if out["frame"].isna().all():
            out["frame"] = explicit_time
    else:
        out["time_s"] = out["frame"] * dt_s
    out["follower_id"] = _series_by_candidates(raw, ["trackId", "track_id"], "follower", is_text=True)
    out["leader_id"] = _series_by_candidates(raw, ["leader_track_id", "leaderId", "leader_id"], "leader", is_text=True)
    out["pair_id"] = dataset_name + "_" + recording.astype(str) + "_" + out["follower_id"].astype(str) + "_" + out["leader_id"].astype(str)

    out["follower_type_raw"] = _series_by_candidates(raw, ["class", "category", "subject_category"], "unknown", is_text=True)
    out["leader_type_raw"] = _series_by_candidates(raw, ["leaderType", "leader_category"], "unknown", is_text=True)
    out["follower_group"] = "Benchmark"
    out["leader_group"] = normalize_benchmark_type(out["leader_type_raw"])
    out["analysis_group"] = "Benchmark overall"
    out["raw_pair_type"] = out["leader_type_raw"] + "-" + out["follower_type_raw"]
    out["follower_is_av"] = False
    out["follower_is_hv"] = False
    out["leader_is_bicycle"] = out["leader_group"].eq("bicycle")

    out["f_x"] = _series_by_candidates(raw, ["xCenter", "translation_x"])
    out["f_y"] = _series_by_candidates(raw, ["yCenter", "translation_y"])
    out["l_x"] = _series_by_candidates(raw, ["leader_xCenter", "leader_translation_x"])
    out["l_y"] = _series_by_candidates(raw, ["leader_yCenter", "leader_translation_y"])
    f_heading = _series_by_candidates(raw, ["heading", "heading_deg"])
    l_heading = _series_by_candidates(raw, ["leader_heading", "leader_heading_deg"])
    out["f_heading"] = f_heading
    out["l_heading"] = l_heading

    f_vx = _series_by_candidates(raw, ["xVelocity", "velocity_x"])
    f_vy = _series_by_candidates(raw, ["yVelocity", "velocity_y"])
    l_vx = _series_by_candidates(raw, ["leader_xVelocity", "leader_velocity_x"])
    l_vy = _series_by_candidates(raw, ["leader_yVelocity", "leader_velocity_y"])
    f_scalar = _series_by_candidates(raw, ["lonVelocity", "v_s"])
    l_scalar = _series_by_candidates(raw, ["leader_lonVelocity", "leader_v_s"])
    out["f_v"] = np.sqrt(f_vx ** 2 + f_vy ** 2).where(f_vx.notna() & f_vy.notna(), f_scalar.abs())
    out["l_v"] = np.sqrt(l_vx ** 2 + l_vy ** 2).where(l_vx.notna() & l_vy.notna(), l_scalar.abs())

    f_lon_acc = _series_by_candidates(raw, ["lonAcceleration"])
    l_lon_acc = _series_by_candidates(raw, ["leader_lonAcceleration"])
    f_ax = _series_by_candidates(raw, ["xAcceleration", "acceleration_x"])
    f_ay = _series_by_candidates(raw, ["yAcceleration", "acceleration_y"])
    l_ax = _series_by_candidates(raw, ["leader_xAcceleration", "leader_acceleration_x"])
    l_ay = _series_by_candidates(raw, ["leader_yAcceleration", "leader_acceleration_y"])
    out["f_a_source"] = f_lon_acc.where(f_lon_acc.notna(), (f_ax * f_vx + f_ay * f_vy) / out["f_v"].where(out["f_v"] > 0.1))
    out["l_a_source"] = l_lon_acc.where(l_lon_acc.notna(), (l_ax * l_vx + l_ay * l_vy) / out["l_v"].where(out["l_v"] > 0.1))

    follower_dim_type = normalize_benchmark_type(out["follower_type_raw"])
    out["f_length"] = _series_by_candidates(raw, ["length", "dimension_x"])
    out["f_width"] = _series_by_candidates(raw, ["width", "dimension_y"])
    out["l_length"] = _series_by_candidates(raw, ["leader_dimension_x"])
    out["l_width"] = _series_by_candidates(raw, ["leader_dimension_y"])
    out["f_length"], out["f_width"] = _fill_dimensions(out["f_length"], out["f_width"], follower_dim_type)
    out["l_length"], out["l_width"] = _fill_dimensions(out["l_length"], out["l_width"], out["leader_group"])

    theta = np.deg2rad(f_heading)
    dx = out["l_x"] - out["f_x"]
    dy = out["l_y"] - out["f_y"]
    half_lengths = 0.5 * (out["f_length"] + out["l_length"])
    out["d_parallel"] = dx * np.cos(theta) + dy * np.sin(theta)
    out["spacing"] = out["d_parallel"] - half_lengths
    out["lateral_offset"] = -dx * np.sin(theta) + dy * np.cos(theta)
    return add_common_metrics(out, preserve_acceleration=True)


def pair_duration_table(df: pd.DataFrame) -> pd.DataFrame:
    if "dt" in df.columns:
        dt = pd.to_numeric(df["dt"], errors="coerce")
    else:
        ordered = df.sort_values(["pair_id", "time_s", "frame"], kind="mergesort")
        dt = ordered.groupby("pair_id", sort=False)["time_s"].diff()
        dt = dt.reindex(df.index)
    temp = df.assign(_dt=dt)
    result = temp.groupby(["dataset_source", "dataset_name", "pair_id"], dropna=False).agg(
        n_frames=("time_s", "size"), start_time=("time_s", "min"), end_time=("time_s", "max"), median_dt=("_dt", "median")
    ).reset_index()
    result["duration_s"] = result["end_time"] - result["start_time"] + result["median_dt"].fillna(0)
    return result


def apply_filters(
    df: pd.DataFrame,
    min_duration_s: float = 5.0,
    max_headway_s: float = 60.0,
    max_lateral_offset_m: float = 2.5,
    remove_near_stationary: bool = False,
    near_stationary_speed: float = 0.5,
) -> pd.DataFrame:
    mask = pd.to_numeric(df["duration_s"], errors="coerce") >= min_duration_s
    mask &= pd.to_numeric(df["spacing"], errors="coerce") > 0
    headway = pd.to_numeric(df["time_headway"], errors="coerce")
    mask &= headway.between(0, max_headway_s)
    lateral = pd.to_numeric(df["lateral_offset"], errors="coerce")
    mask &= lateral.notna() & (lateral.abs() <= max_lateral_offset_m)
    if remove_near_stationary:
        mask &= pd.to_numeric(df["f_v"], errors="coerce") >= near_stationary_speed
    return df.loc[mask].copy()


PRIMARY_METRICS = [
    "spacing", "time_headway", "relative_speed_f_minus_l", "f_a_used", "f_jerk",
    "lateral_offset", "abs_lateral_offset", "mttc", "drac",
]


def pair_level_metrics(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    group_cols = group_cols or ["analysis_group"]
    descriptors = [c for c in group_cols + ["dataset_source", "dataset_name", "follower_group", "leader_group", "raw_pair_type"] if c in df.columns]
    first = df.groupby("pair_id", dropna=False)[descriptors].first()
    agg = df.groupby("pair_id", dropna=False).agg(
        frames=("pair_id", "size"), duration_s=("duration_s", "first"),
        mean_speed=("f_v", "mean"),
        mean_spacing=("spacing", "mean"), median_time_headway=("time_headway", "median"),
        mean_relative_speed=("relative_speed_f_minus_l", "mean"), mean_acceleration=("f_a_used", "mean"),
        std_acceleration=("f_a_used", "std"), mean_jerk=("f_jerk", "mean"), std_jerk=("f_jerk", "std"),
        mean_lateral_offset=("lateral_offset", "mean"), mean_abs_lateral_offset=("abs_lateral_offset", "mean"),
        min_ttc=("ttc", "min"), p_ttc_lt_1p5=("ttc", lambda x: (pd.to_numeric(x, errors="coerce") < 1.5).mean()),
        p_ttc_lt_3=("ttc", lambda x: (pd.to_numeric(x, errors="coerce") < 3.0).mean()),
        min_mttc=("mttc", "min"), mean_drac=("drac", "mean"), max_drac=("drac", "max"),
        p_mttc_lt_1p5=("mttc", lambda x: (pd.to_numeric(x, errors="coerce") < 1.5).mean()),
        p_mttc_lt_3=("mttc", lambda x: (pd.to_numeric(x, errors="coerce") < 3.0).mean()),
        tit_3=("tit_deficit_3p0", lambda x: (x * df.loc[x.index, "dt"]).sum()),
        tit_1p5=("tit_deficit_1p5", lambda x: (x * df.loc[x.index, "dt"]).sum()),
    )
    return first.join(agg).reset_index()
