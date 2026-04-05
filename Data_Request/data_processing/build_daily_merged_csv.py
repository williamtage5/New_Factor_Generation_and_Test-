from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw_request"
OUT_DIR = BASE_DIR / "data" / "merged_data"

OHLC_FILE = RAW_DIR / "请求一段时间内的沪深300股票的OHLC+weight.csv"
VAL_FILE = RAW_DIR / "请求一段时间的沪深300的PEPB市值.csv"
FIN_FILE = RAW_DIR / "请求一段时间内的沪深300的季度财务数据ROE、营收增长率、净利润增长率.csv"


def normalize_yyyymmdd(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d{8})", expand=False)
    return extracted


def assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        count = int(dup.sum())
        sample = df.loc[dup, keys].head(5).to_dict("records")
        raise ValueError(f"{name} has duplicated keys {keys}: {count}, sample={sample}")


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(OHLC_FILE, encoding="utf-8-sig", low_memory=False)
    val = pd.read_csv(VAL_FILE, encoding="utf-8-sig", low_memory=False)
    fin = pd.read_csv(FIN_FILE, encoding="utf-8-sig", low_memory=False)

    base["TradeDate"] = normalize_yyyymmdd(base["TradeDate"])
    val["TradeDate"] = normalize_yyyymmdd(val["TradeDate"])
    fin["AnnounceDate"] = normalize_yyyymmdd(fin["AnnounceDate"])
    fin["ReportPeriod"] = normalize_yyyymmdd(fin["ReportPeriod"])

    base["TradeDate_dt"] = pd.to_datetime(base["TradeDate"], format="%Y%m%d", errors="coerce")
    val["TradeDate_dt"] = pd.to_datetime(val["TradeDate"], format="%Y%m%d", errors="coerce")
    fin["AnnounceDate_dt"] = pd.to_datetime(fin["AnnounceDate"], format="%Y%m%d", errors="coerce")

    base = base.dropna(subset=["StockCode", "TradeDate", "TradeDate_dt"])
    val = val.dropna(subset=["StockCode", "TradeDate", "TradeDate_dt"])
    fin = fin.dropna(subset=["StockCode", "AnnounceDate", "AnnounceDate_dt"])

    assert_unique(base, ["StockCode", "TradeDate"], "OHLC base")
    assert_unique(val, ["StockCode", "TradeDate"], "Valuation")

    merged = base.merge(
        val[["StockCode", "TradeDate", "PE_TTM", "PB", "TotalMarketCap"]],
        on=["StockCode", "TradeDate"],
        how="left",
    )

    merged = merged.sort_values(["StockCode", "TradeDate_dt"]).reset_index(drop=True)
    fin_sorted = fin.sort_values(["StockCode", "AnnounceDate_dt"]).reset_index(drop=True)

    aligned_parts: list[pd.DataFrame] = []
    fin_cols = [
        "AnnounceDate_dt",
        "AnnounceDate",
        "ReportPeriod",
        "ROE",
        "Revenue_YoY",
        "NetProfit_YoY",
    ]
    fin_groups = {k: g for k, g in fin_sorted.groupby("StockCode", sort=False)}

    for stock_code, left_g in merged.groupby("StockCode", sort=False):
        left_g = left_g.sort_values("TradeDate_dt").reset_index(drop=True)
        right_g = fin_groups.get(stock_code)
        if right_g is None or right_g.empty:
            for c in fin_cols:
                left_g[c] = pd.NA
            aligned_parts.append(left_g)
            continue

        right_g = right_g[fin_cols].sort_values("AnnounceDate_dt").reset_index(drop=True)
        aligned = pd.merge_asof(
            left_g,
            right_g,
            left_on="TradeDate_dt",
            right_on="AnnounceDate_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        aligned_parts.append(aligned)

    merged = pd.concat(aligned_parts, axis=0, ignore_index=True)

    leakage_count = int(
        (merged["AnnounceDate_dt"].notna() & (merged["AnnounceDate_dt"] > merged["TradeDate_dt"])).sum()
    )

    final_cols = [
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
    merged = merged[final_cols].copy()

    for old_file in OUT_DIR.glob("*.csv"):
        old_file.unlink()

    for trade_date, day_df in merged.groupby("TradeDate", sort=True):
        out_file = OUT_DIR / f"{trade_date}.csv"
        day_df.sort_values(["StockCode"]).to_csv(out_file, index=False, encoding="utf-8-sig")

    day_files = sorted(OUT_DIR.glob("*.csv"))
    row_counts = [sum(1 for _ in open(p, "r", encoding="utf-8-sig")) - 1 for p in day_files]

    summary = {
        "output_dir": str(OUT_DIR),
        "total_rows": int(len(merged)),
        "total_days": int(len(day_files)),
        "min_trade_date": str(merged["TradeDate"].min()),
        "max_trade_date": str(merged["TradeDate"].max()),
        "min_rows_per_day": int(min(row_counts)) if row_counts else 0,
        "max_rows_per_day": int(max(row_counts)) if row_counts else 0,
        "future_data_leakage_count": leakage_count,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
