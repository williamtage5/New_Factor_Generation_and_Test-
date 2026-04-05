from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_zscore(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    std = x.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean()) / std


def _extract_ic_metrics(ic_decay: pd.DataFrame, ic_lags: list[int]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for lag in ic_lags:
        row = ic_decay.loc[ic_decay["lag"] == lag].head(1)
        out[f"ic_mean_{lag}d"] = float(row["mean_ic"].iloc[0]) if not row.empty else np.nan
        out[f"icir_{lag}d"] = (
            float(row["icir_annualized"].iloc[0]) if not row.empty else np.nan
        )
        out[f"ic_pos_ratio_{lag}d"] = (
            float(row["positive_ratio"].iloc[0]) if not row.empty else np.nan
        )
        out[f"ic_count_{lag}d"] = int(row["count"].iloc[0]) if not row.empty else 0

    ic1 = out.get("ic_mean_1d", np.nan)
    ic20 = out.get("ic_mean_20d", np.nan)
    out["ic_decay_ratio_abs_20_1"] = (
        abs(float(ic20)) / abs(float(ic1))
        if pd.notna(ic1) and pd.notna(ic20) and abs(float(ic1)) > 1e-12
        else np.nan
    )
    out["ic_direction_1d"] = (
        "positive"
        if pd.notna(ic1) and float(ic1) > 0
        else ("negative" if pd.notna(ic1) and float(ic1) < 0 else "flat")
    )
    return out


def _extract_ls_metrics(group_returns_stats: pd.DataFrame) -> dict[str, float | int]:
    row = group_returns_stats.loc[group_returns_stats["group"] == "LS_Gn_G1"].head(1)
    if row.empty:
        return {
            "ls_count": 0,
            "ls_mean_daily": np.nan,
            "ls_annual_return": np.nan,
            "ls_annual_vol": np.nan,
            "ls_ir": np.nan,
            "ls_win_rate": np.nan,
            "ls_max_drawdown": np.nan,
            "ls_cum_return": np.nan,
        }
    return {
        "ls_count": int(row["count"].iloc[0]),
        "ls_mean_daily": float(row["mean_daily"].iloc[0]),
        "ls_annual_return": float(row["annual_return"].iloc[0]),
        "ls_annual_vol": float(row["annual_vol"].iloc[0]),
        "ls_ir": float(row["ir"].iloc[0]),
        "ls_win_rate": float(row["win_rate"].iloc[0]),
        "ls_max_drawdown": float(row["max_drawdown"].iloc[0]),
        "ls_cum_return": float(row["cum_return"].iloc[0]),
    }


def _extract_distribution_metrics(dist_df: pd.DataFrame) -> dict[str, float]:
    row = dist_df.head(1)
    if row.empty:
        return {
            "dist_mean": np.nan,
            "dist_std": np.nan,
            "dist_skew": np.nan,
            "dist_kurtosis": np.nan,
        }
    return {
        "dist_mean": float(row["mean"].iloc[0]),
        "dist_std": float(row["std"].iloc[0]),
        "dist_skew": float(row["skew"].iloc[0]),
        "dist_kurtosis": float(row["kurtosis"].iloc[0]),
    }


def _extract_ls_yearly(group_returns_daily: pd.DataFrame, factor: str) -> pd.DataFrame:
    if group_returns_daily.empty or "LS_Gn_G1" not in group_returns_daily.columns:
        return pd.DataFrame(columns=["factor", "year", "ls_mean_daily"])

    tmp = group_returns_daily.copy()
    tmp["TradeDate"] = pd.to_datetime(tmp["TradeDate"], errors="coerce")
    tmp = tmp.dropna(subset=["TradeDate"])
    tmp["year"] = tmp["TradeDate"].dt.year
    tmp["LS_Gn_G1"] = pd.to_numeric(tmp["LS_Gn_G1"], errors="coerce")
    yearly = (
        tmp.groupby("year", as_index=False)["LS_Gn_G1"]
        .mean()
        .rename(columns={"LS_Gn_G1": "ls_mean_daily"})
    )
    yearly["factor"] = factor
    return yearly[["factor", "year", "ls_mean_daily"]]


def build_master_summary(
    factor_data: dict[str, dict[str, pd.DataFrame]], ic_lags: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, float | int | str]] = []
    yearly_rows: list[pd.DataFrame] = []

    for factor, data in sorted(factor_data.items()):
        ic_metrics = _extract_ic_metrics(data["ic_decay"], ic_lags=ic_lags)
        ls_metrics = _extract_ls_metrics(data["group_returns_stats"])
        dist_metrics = _extract_distribution_metrics(data["factor_distribution_stats"])

        row: dict[str, float | int | str] = {"factor": factor}
        row.update(ic_metrics)
        row.update(ls_metrics)
        row.update(dist_metrics)
        summary_rows.append(row)

        yearly_rows.append(_extract_ls_yearly(data["group_returns_daily"], factor))

    summary = pd.DataFrame(summary_rows).sort_values("factor").reset_index(drop=True)

    summary["score_ic_strength"] = _safe_zscore(summary["icir_1d"].abs())
    summary["score_ls_strength"] = _safe_zscore(summary["ls_ir"].abs())
    summary["score_cum_strength"] = _safe_zscore(summary["ls_cum_return"].abs())
    summary["score_total"] = (
        summary["score_ic_strength"]
        + summary["score_ls_strength"]
        + summary["score_cum_strength"]
    )

    ic_cols = [f"ic_mean_{lag}d" for lag in ic_lags]
    ic_lag_matrix = summary[["factor"] + ic_cols].copy()
    ic_lag_matrix.columns = ["factor"] + [f"lag_{lag}d" for lag in ic_lags]

    yearly_all = (
        pd.concat(yearly_rows, ignore_index=True)
        if yearly_rows
        else pd.DataFrame(columns=["factor", "year", "ls_mean_daily"])
    )
    ls_yearly_matrix = yearly_all.pivot_table(
        index="factor", columns="year", values="ls_mean_daily", aggfunc="mean"
    )
    ls_yearly_matrix = ls_yearly_matrix.sort_index()
    ls_yearly_matrix.columns = [str(c) for c in ls_yearly_matrix.columns]
    ls_yearly_matrix = ls_yearly_matrix.reset_index()

    return summary, ic_lag_matrix, ls_yearly_matrix


def build_rank_tables(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rank_ic = summary.sort_values("icir_1d", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    rank_ls = summary.sort_values("ls_ir", ascending=False).reset_index(drop=True)
    rank_composite = summary.sort_values("score_total", ascending=False).reset_index(drop=True)
    return rank_ic, rank_ls, rank_composite


def build_quality_flags(
    summary: pd.DataFrame,
    icir_abs_threshold: float,
    ls_ir_abs_threshold: float,
    ls_win_rate_threshold: float,
) -> pd.DataFrame:
    flags = summary[["factor", "icir_1d", "ls_ir", "ls_win_rate"]].copy()
    flags["pass_icir_abs"] = flags["icir_1d"].abs() >= icir_abs_threshold
    flags["pass_ls_ir_abs"] = flags["ls_ir"].abs() >= ls_ir_abs_threshold
    flags["pass_ls_win_rate"] = flags["ls_win_rate"] >= ls_win_rate_threshold
    flags["pass_all"] = (
        flags["pass_icir_abs"] & flags["pass_ls_ir_abs"] & flags["pass_ls_win_rate"]
    )
    return flags.sort_values(["pass_all", "factor"], ascending=[False, True]).reset_index(
        drop=True
    )
