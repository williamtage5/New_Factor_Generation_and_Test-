from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "remove_NaN"
OUTPUT_DIR = BASE_DIR / "data" / "factor_generation"


def fill_leading_with_first_valid(s: pd.Series) -> pd.Series:
    """Fill leading NaN values with the first valid value in the same stock series."""
    out = s.copy()
    first_idx = out.first_valid_index()
    if first_idx is None:
        return out.fillna(0.0)
    first_pos = out.index.get_loc(first_idx)
    if isinstance(first_pos, slice):
        first_pos = first_pos.start
    out.iloc[:first_pos] = out.iloc[first_pos]
    return out


def rolling_by_stock(
    df: pd.DataFrame, value_col: str, window: int, func: str
) -> pd.Series:
    g = df.groupby("StockCode", sort=False)[value_col]
    if func == "mean":
        return g.transform(lambda x: x.rolling(window, min_periods=window).mean())
    if func == "std":
        return g.transform(lambda x: x.rolling(window, min_periods=window).std())
    if func == "max":
        return g.transform(lambda x: x.rolling(window, min_periods=window).max())
    raise ValueError(f"Unsupported rolling func: {func}")


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No input files found in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in OUTPUT_DIR.glob("*.csv"):
        f.unlink()

    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in files],
        ignore_index=True,
    )
    df["TradeDate"] = df["TradeDate"].astype(str).str.extract(r"(\d{8})", expand=False)
    df = df.sort_values(["StockCode", "TradeDate"]).reset_index(drop=True)

    numeric_cols = [
        "ClosePrice",
        "Volume",
        "Amount",
        "TotalMarketCap",
        "PE_TTM",
        "PB",
        "ROE",
        "Revenue_YoY",
        "NetProfit_YoY",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Base return
    df["_ret_1d"] = df.groupby("StockCode", sort=False)["ClosePrice"].pct_change()

    # Factor construction (raw)
    df["momentum_5_raw"] = df.groupby("StockCode", sort=False)["ClosePrice"].pct_change(5)
    df["momentum_20_raw"] = df.groupby("StockCode", sort=False)["ClosePrice"].pct_change(20)
    df["momentum_60_raw"] = df.groupby("StockCode", sort=False)["ClosePrice"].pct_change(60)
    df["reversal_5_raw"] = -df["momentum_5_raw"]

    df["volatility_20_raw"] = rolling_by_stock(df, "_ret_1d", 20, "std")
    df["_ret_neg"] = df["_ret_1d"].where(df["_ret_1d"] < 0, 0.0)
    df["downside_vol_20_raw"] = rolling_by_stock(df, "_ret_neg", 20, "std")

    rolling_max_20 = rolling_by_stock(df, "ClosePrice", 20, "max")
    df["drawdown_20_raw"] = df["ClosePrice"] / rolling_max_20 - 1.0

    df["_turnover_proxy_1d"] = np.where(
        df["TotalMarketCap"].abs() > 1e-12,
        df["Amount"] / df["TotalMarketCap"],
        0.0,
    )
    df["turnover_proxy_20_raw"] = rolling_by_stock(df, "_turnover_proxy_1d", 20, "mean")

    df["_illiq_1d"] = np.abs(df["_ret_1d"]) / (df["Amount"].abs() + 1.0)
    df["illiq_20_raw"] = rolling_by_stock(df, "_illiq_1d", 20, "mean")

    df["roe_raw"] = df["ROE"]
    df["size_log_mcap_raw"] = np.log(np.clip(df["TotalMarketCap"], 1.0, None))
    df["ep_ttm_raw"] = np.where(df["PE_TTM"].abs() > 1e-12, 1.0 / df["PE_TTM"], 0.0)
    df["bp_inv_raw"] = np.where(df["PB"].abs() > 1e-12, 1.0 / df["PB"], 0.0)
    df["revenue_yoy_raw"] = df["Revenue_YoY"]
    df["netprofit_yoy_raw"] = df["NetProfit_YoY"]

    factor_cols = [
        "momentum_5_raw",
        "momentum_20_raw",
        "momentum_60_raw",
        "reversal_5_raw",
        "volatility_20_raw",
        "downside_vol_20_raw",
        "drawdown_20_raw",
        "turnover_proxy_20_raw",
        "illiq_20_raw",
        "roe_raw",
        "size_log_mcap_raw",
        "ep_ttm_raw",
        "bp_inv_raw",
        "revenue_yoy_raw",
        "netprofit_yoy_raw",
    ]

    missing_before_fill = {c: int(df[c].isna().sum()) for c in factor_cols}

    rolling_cols = [
        "momentum_5_raw",
        "momentum_20_raw",
        "momentum_60_raw",
        "reversal_5_raw",
        "volatility_20_raw",
        "downside_vol_20_raw",
        "drawdown_20_raw",
        "turnover_proxy_20_raw",
        "illiq_20_raw",
    ]

    # User requirement: fill warm-up NaN with each stock's first valid value.
    for c in rolling_cols:
        df[c] = (
            df.groupby("StockCode", sort=False)[c]
            .transform(fill_leading_with_first_valid)
        )

    # Safety: replace inf and fill any remaining NaN by stock ffill+bfill, then 0.
    df[factor_cols] = df[factor_cols].replace([np.inf, -np.inf], np.nan)
    for c in factor_cols:
        df[c] = (
            df.groupby("StockCode", sort=False)[c]
            .transform(lambda s: s.ffill().bfill())
            .fillna(0.0)
        )

    missing_after_fill = {c: int(df[c].isna().sum()) for c in factor_cols}

    base_cols = [
        "TradeDate",
        "StockCode",
        "IndexWeight",
        "ClosePrice",
        "Volume",
        "Amount",
        "PE_TTM",
        "PB",
        "TotalMarketCap",
        "AnnounceDate",
        "ReportPeriod",
        "ROE",
        "Revenue_YoY",
        "NetProfit_YoY",
    ]
    out_cols = base_cols + factor_cols
    out_df = df[out_cols].sort_values(["TradeDate", "StockCode"]).reset_index(drop=True)

    for td, day_df in out_df.groupby("TradeDate", sort=True):
        out_file = OUTPUT_DIR / f"{td}.csv"
        day_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    file_count = len(list(OUTPUT_DIR.glob("*.csv")))
    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "rows": int(len(out_df)),
        "days": int(out_df["TradeDate"].nunique()),
        "start_date": str(out_df["TradeDate"].min()),
        "end_date": str(out_df["TradeDate"].max()),
        "files_written": file_count,
        "factor_count": len(factor_cols),
        "factor_columns": factor_cols,
        "missing_before_fill": missing_before_fill,
        "missing_after_fill": missing_after_fill,
        "total_factor_missing_after_fill": int(out_df[factor_cols].isna().sum().sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

