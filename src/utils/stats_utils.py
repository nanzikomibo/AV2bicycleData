from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def summarize_numeric(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for metric in metrics:
            x = pd.to_numeric(group.get(metric), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            row = base | {"metric": metric, "n": int(len(x))}
            if len(x):
                row |= {
                    "mean": x.mean(), "median": x.median(), "std": x.std(),
                    "p05": x.quantile(0.05), "p25": x.quantile(0.25),
                    "p75": x.quantile(0.75), "p95": x.quantile(0.95),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def trajectory_quality_summary(df: pd.DataFrame, group_col: str, jerk_threshold: float = 15.0) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(group_col, dropna=False):
        speed = pd.to_numeric(group["f_v"], errors="coerce")
        speed_pos = pd.to_numeric(group["f_speed_from_pos"], errors="coerce")
        acc = pd.to_numeric(group["f_a_used"], errors="coerce")
        jerk = pd.to_numeric(group["f_jerk"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        spacing = pd.to_numeric(group["spacing"], errors="coerce")
        headway = pd.to_numeric(group["time_headway"], errors="coerce")
        lateral = pd.to_numeric(group["lateral_offset"], errors="coerce")
        rows.append({
            group_col: name,
            "n_frames": len(group), "n_pairs": group["pair_id"].nunique(),
            "speed_position_rmse": np.sqrt(np.nanmean((speed_pos - speed) ** 2)),
            "acceleration_mean": acc.mean(), "acceleration_std": acc.std(),
            "jerk_mean": jerk.mean(), "jerk_std": jerk.std(),
            "abnormal_jerk_ratio": (jerk.abs() > jerk_threshold).mean(),
            "invalid_spacing_ratio": (spacing.isna() | (spacing <= 0)).mean(),
            "invalid_headway_ratio": (headway.isna() | (headway < 0) | (headway > 60)).mean(),
            "lateral_offset_missing_ratio": lateral.isna().mean(),
        })
    return pd.DataFrame(rows)


def tail_probabilities(df: pd.DataFrame, group_col: str, metric: str = "mttc") -> pd.DataFrame:
    """Threshold probabilities use all valid frames; infinity remains in the denominator."""
    rows = []
    for name, group in df.groupby(group_col, dropna=False):
        x = pd.to_numeric(group[metric], errors="coerce").dropna()
        finite = x.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            group_col: name, "metric": metric, "n_all_valid": len(x), "n_finite": len(finite),
            "mean_finite": finite.mean(), "median_finite": finite.median(),
            "p05_finite": finite.quantile(0.05), "p10_finite": finite.quantile(0.10),
            f"p_{metric}_lt_1": (x < 1.0).mean(), f"p_{metric}_lt_1p5": (x < 1.5).mean(),
            f"p_{metric}_lt_3": (x < 3.0).mean(), "p_no_closing_or_infinite": np.isinf(x).mean(),
        })
    return pd.DataFrame(rows)


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    values = p[valid].to_numpy()
    if not len(values):
        return pd.Series(np.nan, index=p.index)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = pd.Series(np.nan, index=p.index)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result.loc[valid] = restored
    return result


def two_group_tests(df: pd.DataFrame, group_col: str, group_a: str, group_b: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        a = pd.to_numeric(df.loc[df[group_col] == group_a, metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        b = pd.to_numeric(df.loc[df[group_col] == group_b, metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        row = {
            "metric": metric, "group_a": group_a, "group_b": group_b,
            "n_a": len(a), "n_b": len(b), "mean_a": a.mean(), "mean_b": b.mean(),
            "median_a": a.median(), "median_b": b.median(),
        }
        if len(a) >= 3 and len(b) >= 3:
            mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            row["mann_whitney_u"] = mw.statistic
            row["mann_whitney_p"] = mw.pvalue
            row["rank_biserial"] = 2 * mw.statistic / (len(a) * len(b)) - 1
            row["ks_statistic"] = stats.ks_2samp(a, b).statistic
            row["ks_p"] = stats.ks_2samp(a, b).pvalue
            pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
            row["cohens_d"] = (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)
    if "mann_whitney_p" in result:
        result["mann_whitney_p_fdr_bh"] = _bh_adjust(result["mann_whitney_p"])
    if "ks_p" in result:
        result["ks_p_fdr_bh"] = _bh_adjust(result["ks_p"])
    return result


def low_mttc_pair_test(pair_df: pd.DataFrame, group_col: str, group_a: str, group_b: str, threshold: float = 1.5) -> pd.DataFrame:
    a = pair_df.loc[pair_df[group_col] == group_a, "min_mttc"].lt(threshold)
    b = pair_df.loc[pair_df[group_col] == group_b, "min_mttc"].lt(threshold)
    if len(a) and len(b):
        table = np.array([[a.sum(), len(a) - a.sum()], [b.sum(), len(b) - b.sum()]])
        _, p = stats.fisher_exact(table)
    else:
        p = np.nan
    return pd.DataFrame([{
        "group_a": group_a, "group_b": group_b, "threshold_s": threshold,
        "n_pairs_a": len(a), "n_pairs_b": len(b), "share_pairs_a": a.mean(),
        "share_pairs_b": b.mean(), "fisher_exact_p": p,
    }])


def ols_hc3(df: pd.DataFrame, dependent_variables: list[str], level: str) -> pd.DataFrame:
    import statsmodels.api as sm

    rows = []
    base_predictors = ["leader_bicycle", "follower_av", "leader_bicycle_x_follower_av"]
    for y in dependent_variables:
        predictors = base_predictors.copy()
        if y != "mean_speed" and "mean_speed" in df.columns:
            predictors.append("mean_speed")
        columns = list(dict.fromkeys([y] + predictors))
        data = df[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 30 or data[y].nunique() < 3:
            rows.append({"level": level, "dependent_variable": y, "error": "insufficient observations", "n": len(data)})
            continue
        x = sm.add_constant(data[predictors], has_constant="add")
        model = sm.OLS(data[y], x).fit(cov_type="HC3")
        for term in model.params.index:
            rows.append({
                "level": level, "dependent_variable": y, "term": term,
                "coef": model.params[term], "std_err_hc3": model.bse[term],
                "t_value": model.tvalues[term], "p_value": model.pvalues[term],
                "ci95_low": model.conf_int().loc[term, 0], "ci95_high": model.conf_int().loc[term, 1],
                "n": int(model.nobs), "r2": model.rsquared, "adjusted_r2": model.rsquared_adj,
            })
    return pd.DataFrame(rows)
