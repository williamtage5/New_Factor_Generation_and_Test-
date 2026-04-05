from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "factor_generation"
OUTPUT_DIR = BASE_DIR / "data" / "handled_outliers"

FACTOR_COLS = [
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

SIGMA_K = 3.0
MIN_COUNT = 10


def main() -> None:
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT_DIR.glob("*.csv"):
        old.unlink()

    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in files],
        ignore_index=True,
    )
    df["TradeDate"] = df["TradeDate"].astype(str).str.extract(r"(\d{8})", expand=False)
    df = df.sort_values(["TradeDate", "StockCode"]).reset_index(drop=True)

    factor_stats: dict[str, dict[str, float | int]] = {}

    for col in FACTOR_COLS:
        if col not in df.columns:
            raise KeyError(f"Missing factor column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

        mu = df.groupby("TradeDate")[col].transform("mean")
        sigma = df.groupby("TradeDate")[col].transform("std")
        cnt = df.groupby("TradeDate")[col].transform("count")

        lower = mu - SIGMA_K * sigma
        upper = mu + SIGMA_K * sigma

        skip_mask = sigma.isna() | (sigma <= 0) | (cnt < MIN_COUNT)
        active_mask = ~skip_mask

        x = df[col]
        clip_hi = active_mask & (x > upper)
        clip_lo = active_mask & (x < lower)
        clip_mask = clip_hi | clip_lo

        clipped = x.where(~active_mask, x.clip(lower=lower, upper=upper))
        out_col = f"{col}_clip3"
        df[out_col] = clipped

        adj = (x - clipped).abs()
        factor_stats[col] = {
            "rows_total": int(len(df)),
            "rows_skipped_sigma_rule": int(skip_mask.sum()),
            "rows_clipped_total": int(clip_mask.sum()),
            "rows_clipped_upper": int(clip_hi.sum()),
            "rows_clipped_lower": int(clip_lo.sum()),
            "clipped_pct": round(float(clip_mask.mean() * 100.0), 6),
            "max_abs_adjustment": float(adj.max(skipna=True) if adj.notna().any() else 0.0),
        }

    out_cols = list(df.columns)
    out_df = df[out_cols].sort_values(["TradeDate", "StockCode"]).reset_index(drop=True)

    for td, day_df in out_df.groupby("TradeDate", sort=True):
        out_file = OUTPUT_DIR / f"{td}.csv"
        day_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    out_files = sorted(OUTPUT_DIR.glob("*.csv"))
    bad_daily_rows = []
    for f in out_files:
        n = sum(1 for _ in open(f, "r", encoding="utf-8-sig")) - 1
        if n != 300:
            bad_daily_rows.append((f.stem, n))

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "rows_total": int(len(out_df)),
        "days_total": int(out_df["TradeDate"].nunique()),
        "start_date": str(out_df["TradeDate"].min()),
        "end_date": str(out_df["TradeDate"].max()),
        "files_written": len(out_files),
        "sigma_k": SIGMA_K,
        "min_count_rule": MIN_COUNT,
        "total_missing_values": int(out_df.isna().sum().sum()),
        "daily_rows_not_300_count": len(bad_daily_rows),
        "daily_rows_not_300_sample": bad_daily_rows[:10],
        "factor_stats": factor_stats,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

