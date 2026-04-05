from __future__ import annotations

import numpy as np
import pandas as pd


def daily_spearman(
    df_day: pd.DataFrame, factor_col: str, ret_col: str, min_samples: int
) -> float:
    sub = df_day[[factor_col, ret_col]].dropna()
    if len(sub) < min_samples:
        return np.nan
    if sub[factor_col].nunique(dropna=True) <= 1 or sub[ret_col].nunique(dropna=True) <= 1:
        return np.nan
    return sub[factor_col].rank(method="average").corr(
        sub[ret_col].rank(method="average")
    )


def compute_ic_series(
    panel: pd.DataFrame,
    factor_col: str,
    ic_lags: list[int],
    min_ic_samples: int,
) -> pd.DataFrame:
    out = pd.DataFrame({"TradeDate": sorted(panel["TradeDate"].unique())})
    grouped = list(panel.groupby("TradeDate", sort=True))

    for lag in ic_lags:
        ret_col = f"fwd_ret_{lag}d"
        values = [
            daily_spearman(
                day_df,
                factor_col=factor_col,
                ret_col=ret_col,
                min_samples=min_ic_samples,
            )
            for _, day_df in grouped
        ]
        out[f"IC_{lag}d"] = values
    return out


def summarize_ic_decay(ic_df: pd.DataFrame, ic_lags: list[int]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for lag in ic_lags:
        col = f"IC_{lag}d"
        s = pd.to_numeric(ic_df[col], errors="coerce").dropna()
        if s.empty:
            rows.append(
                {
                    "lag": lag,
                    "count": 0,
                    "mean_ic": np.nan,
                    "std_ic": np.nan,
                    "icir_annualized": np.nan,
                    "positive_ratio": np.nan,
                }
            )
            continue

        std = float(s.std(ddof=1))
        icir = float(s.mean() / std * np.sqrt(252.0 / lag)) if std > 0 else np.nan
        rows.append(
            {
                "lag": lag,
                "count": int(len(s)),
                "mean_ic": float(s.mean()),
                "std_ic": std,
                "icir_annualized": icir,
                "positive_ratio": float((s > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def assign_groups(values: pd.Series, n_groups: int) -> pd.Series:
    if len(values) < n_groups:
        return pd.Series(np.nan, index=values.index)
    ranked = values.rank(method="first")
    return pd.qcut(ranked, q=n_groups, labels=False) + 1


def compute_group_returns_daily(
    panel: pd.DataFrame,
    factor_col: str,
    n_groups: int,
    min_group_samples: int,
) -> pd.DataFrame:
    ret_col = "fwd_ret_1d"
    records: list[dict[str, float | str]] = []
    for trade_date, day_df in panel.groupby("TradeDate", sort=True):
        sub = day_df[[factor_col, ret_col]].dropna()
        if len(sub) < max(n_groups, min_group_samples):
            continue

        sub = sub.copy()
        sub["group"] = assign_groups(sub[factor_col], n_groups=n_groups)
        sub = sub.dropna(subset=["group"])
        sub["group"] = sub["group"].astype(int)

        grp_ret = sub.groupby("group", sort=True)[ret_col].mean()
        if len(grp_ret) != n_groups:
            continue

        row: dict[str, float | str] = {"TradeDate": trade_date.strftime("%Y-%m-%d")}
        for g in range(1, n_groups + 1):
            row[f"G{g}"] = float(grp_ret.loc[g])
        row["LS_Gn_G1"] = float(row[f"G{n_groups}"] - row["G1"])
        records.append(row)

    if not records:
        cols = ["TradeDate"] + [f"G{i}" for i in range(1, n_groups + 1)] + ["LS_Gn_G1"]
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(records)


def max_drawdown(ret: pd.Series) -> float:
    if ret.empty:
        return np.nan
    nav = (1.0 + ret.fillna(0.0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def summarize_group_returns(group_daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    if group_daily.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "count",
                "mean_daily",
                "annual_return",
                "annual_vol",
                "ir",
                "win_rate",
                "max_drawdown",
                "cum_return",
            ]
        )

    for col in [c for c in group_daily.columns if c != "TradeDate"]:
        s = pd.to_numeric(group_daily[col], errors="coerce").dropna()
        if s.empty:
            continue
        std = float(s.std(ddof=1))
        ann_vol = std * np.sqrt(252.0)
        ir = float(s.mean() / std * np.sqrt(252.0)) if std > 0 else np.nan
        rows.append(
            {
                "group": col,
                "count": int(len(s)),
                "mean_daily": float(s.mean()),
                "annual_return": float((1.0 + s.mean()) ** 252 - 1.0),
                "annual_vol": float(ann_vol),
                "ir": ir,
                "win_rate": float((s > 0).mean()),
                "max_drawdown": max_drawdown(s),
                "cum_return": float((1.0 + s).prod() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_distribution(panel: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    s = pd.to_numeric(panel[factor_col], errors="coerce").dropna()
    if s.empty:
        return pd.DataFrame(
            [
                {
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p1": np.nan,
                    "p5": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "p95": np.nan,
                    "p99": np.nan,
                    "max": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                }
            ]
        )

    return pd.DataFrame(
        [
            {
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "p1": float(s.quantile(0.01)),
                "p5": float(s.quantile(0.05)),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "skew": float(s.skew()),
                "kurtosis": float(s.kurt()),
            }
        ]
    )
