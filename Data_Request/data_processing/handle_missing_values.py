from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "data" / "merged_data"
OUTPUT_DIR = BASE_DIR / "data" / "remove_NaN"

COVERAGE_THRESHOLD = 0.95
STABLE_DAYS = 3


def load_daily_csvs(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    frames = [pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["TradeDate"] = df["TradeDate"].astype(str).str.extract(r"(\d{8})", expand=False)
    df["AnnounceDate"] = df["AnnounceDate"].astype(str).str.extract(r"(\d{8})", expand=False)
    return df


def find_cutoff_date(daily_coverage: pd.Series, threshold: float, stable_days: int) -> str:
    values = daily_coverage.values
    dates = daily_coverage.index.tolist()
    for i in range(len(values) - stable_days + 1):
        window = values[i : i + stable_days]
        if (window >= threshold).all():
            return str(dates[i])
    raise ValueError(
        f"Cannot find cutoff date with coverage >= {threshold:.2%} for {stable_days} consecutive trading days."
    )


def get_missing_counts(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in cols}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_csv in OUTPUT_DIR.glob("*.csv"):
        old_csv.unlink()

    df = load_daily_csvs(INPUT_DIR)
    # Remove legacy marker columns if they exist.
    legacy_cols = [c for c in ["is_fin_unavailable", "is_rev_yoy_imputed", "is_np_yoy_imputed"] if c in df.columns]
    if legacy_cols:
        df = df.drop(columns=legacy_cols)

    numeric_cols = [
        "IndexWeight",
        "ClosePrice",
        "Volume",
        "Amount",
        "PE_TTM",
        "PB",
        "TotalMarketCap",
        "ROE",
        "Revenue_YoY",
        "NetProfit_YoY",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    input_rows = int(len(df))
    input_days = int(df["TradeDate"].nunique())
    missing_before_all = {k: int(v) for k, v in df.isna().sum().items() if int(v) > 0}

    daily_coverage = (
        df.groupby("TradeDate")["AnnounceDate"]
        .apply(lambda s: float(s.notna().mean()))
        .sort_index()
    )

    cutoff_date = find_cutoff_date(daily_coverage, COVERAGE_THRESHOLD, STABLE_DAYS)
    keep_mask = df["TradeDate"] >= cutoff_date
    df_kept = df.loc[keep_mask].copy()

    dropped_rows = int((~keep_mask).sum())
    dropped_days = int(df.loc[~keep_mask, "TradeDate"].nunique())

    # Keep time order for ffill rules.
    df_kept = df_kept.sort_values(["StockCode", "TradeDate"]).reset_index(drop=True)

    step_stats: dict[str, object] = {}
    step_stats["after_cutoff_missing"] = {k: int(v) for k, v in df_kept.isna().sum().items() if int(v) > 0}

    # ================= Step A: Market columns =================
    market_cols = ["ClosePrice", "Volume", "Amount"]
    step_a_before = get_missing_counts(df_kept, market_cols)

    all_three_missing = df_kept["ClosePrice"].isna() & df_kept["Volume"].isna() & df_kept["Amount"].isna()
    n_all_three = int(all_three_missing.sum())

    # Rule A1: all missing -> ClosePrice ffill by stock; Volume=0; Amount=0
    close_ff = df_kept.groupby("StockCode", sort=False)["ClosePrice"].transform("ffill")
    fill_close_all_three = all_three_missing & df_kept["ClosePrice"].isna() & close_ff.notna()
    df_kept.loc[fill_close_all_three, "ClosePrice"] = close_ff[fill_close_all_three]

    # Fallback for remaining close missing: daily median then global median
    miss_close = all_three_missing & df_kept["ClosePrice"].isna()
    if miss_close.any():
        med_daily_close = df_kept.groupby("TradeDate")["ClosePrice"].transform("median")
        fill_daily = miss_close & med_daily_close.notna()
        df_kept.loc[fill_daily, "ClosePrice"] = med_daily_close[fill_daily]
    miss_close2 = all_three_missing & df_kept["ClosePrice"].isna()
    if miss_close2.any():
        df_kept.loc[miss_close2, "ClosePrice"] = df_kept["ClosePrice"].median()

    vol_missing_before_zero = int(df_kept["Volume"].isna().sum())
    amt_missing_before_zero = int(df_kept["Amount"].isna().sum())
    df_kept.loc[df_kept["Volume"].isna(), "Volume"] = 0.0
    df_kept.loc[df_kept["Amount"].isna(), "Amount"] = 0.0

    # Rule A3: if ClosePrice still missing -> stock ffill, then daily median, then global median
    close_missing_before_rule3 = int(df_kept["ClosePrice"].isna().sum())
    close_ff2 = df_kept.groupby("StockCode", sort=False)["ClosePrice"].transform("ffill")
    mask = df_kept["ClosePrice"].isna() & close_ff2.notna()
    df_kept.loc[mask, "ClosePrice"] = close_ff2[mask]
    miss_close3 = df_kept["ClosePrice"].isna()
    if miss_close3.any():
        med_daily_close2 = df_kept.groupby("TradeDate")["ClosePrice"].transform("median")
        mask2 = miss_close3 & med_daily_close2.notna()
        df_kept.loc[mask2, "ClosePrice"] = med_daily_close2[mask2]
    miss_close4 = df_kept["ClosePrice"].isna()
    if miss_close4.any():
        df_kept.loc[miss_close4, "ClosePrice"] = df_kept["ClosePrice"].median()

    step_a_after = get_missing_counts(df_kept, market_cols)
    step_stats["step_market"] = {
        "before_missing": step_a_before,
        "all_three_missing_rows": n_all_three,
        "volume_filled_to_zero": vol_missing_before_zero,
        "amount_filled_to_zero": amt_missing_before_zero,
        "close_missing_before_rule3": close_missing_before_rule3,
        "after_missing": step_a_after,
    }

    # ================= Step B: Valuation columns =================
    val_cols = ["PE_TTM", "PB", "TotalMarketCap"]
    step_b_before = get_missing_counts(df_kept, val_cols)
    step_b_detail: dict[str, dict[str, int]] = {}
    for col in val_cols:
        miss0 = df_kept[col].isna()
        s_ff = df_kept.groupby("StockCode", sort=False)[col].transform(lambda x: x.ffill(limit=20))
        m1 = miss0 & s_ff.notna()
        df_kept.loc[m1, col] = s_ff[m1]

        miss1 = df_kept[col].isna()
        daily_med = df_kept.groupby("TradeDate")[col].transform("median")
        m2 = miss1 & daily_med.notna()
        df_kept.loc[m2, col] = daily_med[m2]

        miss2 = df_kept[col].isna()
        if miss2.any():
            global_med = df_kept[col].median()
            df_kept.loc[miss2, col] = global_med

        step_b_detail[col] = {
            "filled_by_ffill_limit20": int(m1.sum()),
            "filled_by_daily_median": int(m2.sum()),
            "filled_by_global_median": int(miss2.sum()),
            "remaining_missing": int(df_kept[col].isna().sum()),
        }
    step_b_after = get_missing_counts(df_kept, val_cols)
    step_stats["step_valuation"] = {
        "before_missing": step_b_before,
        "detail": step_b_detail,
        "after_missing": step_b_after,
    }

    # ================= Step C: Financial factor columns =================
    fin_cols = ["ROE", "Revenue_YoY", "NetProfit_YoY"]
    step_c_before = get_missing_counts(df_kept, fin_cols)
    step_c_detail: dict[str, dict[str, int]] = {}
    for col in fin_cols:
        miss0 = df_kept[col].isna()
        s_ff = df_kept.groupby("StockCode", sort=False)[col].transform("ffill")
        m1 = miss0 & s_ff.notna()
        df_kept.loc[m1, col] = s_ff[m1]

        miss1 = df_kept[col].isna()
        daily_med = df_kept.groupby("TradeDate")[col].transform("median")
        m2 = miss1 & daily_med.notna()
        df_kept.loc[m2, col] = daily_med[m2]

        miss2 = df_kept[col].isna()
        if miss2.any():
            global_med = df_kept[col].median()
            df_kept.loc[miss2, col] = global_med

        step_c_detail[col] = {
            "filled_by_ffill": int(m1.sum()),
            "filled_by_daily_median": int(m2.sum()),
            "filled_by_global_median": int(miss2.sum()),
            "remaining_missing": int(df_kept[col].isna().sum()),
        }
    step_c_after = get_missing_counts(df_kept, fin_cols)
    step_stats["step_financial"] = {
        "before_missing": step_c_before,
        "detail": step_c_detail,
        "after_missing": step_c_after,
    }

    # ================= Step D: Date columns =================
    date_cols = ["AnnounceDate", "ReportPeriod"]
    step_d_before = get_missing_counts(df_kept, date_cols)

    # User rule:
    # 1) AnnounceDate missing -> TradeDate
    # 2) ReportPeriod missing -> AnnounceDate
    ann_missing_mask = df_kept["AnnounceDate"].isna()
    ann_filled = int(ann_missing_mask.sum())
    df_kept.loc[ann_missing_mask, "AnnounceDate"] = df_kept.loc[ann_missing_mask, "TradeDate"]

    rep_missing_mask = df_kept["ReportPeriod"].isna()
    rep_filled = int(rep_missing_mask.sum())
    df_kept.loc[rep_missing_mask, "ReportPeriod"] = df_kept.loc[rep_missing_mask, "AnnounceDate"]

    step_d_after = get_missing_counts(df_kept, date_cols)
    step_stats["step_dates"] = {
        "before_missing": step_d_before,
        "announce_filled_with_trade_date": ann_filled,
        "report_period_filled_with_announce_date": rep_filled,
        "after_missing": step_d_after,
    }

    remaining_missing = {k: int(v) for k, v in df_kept.isna().sum().items() if int(v) > 0}

    df_kept = df_kept.sort_values(["TradeDate", "StockCode"]).reset_index(drop=True)
    for trade_date, day_df in df_kept.groupby("TradeDate", sort=True):
        out_file = OUTPUT_DIR / f"{trade_date}.csv"
        day_df.to_csv(out_file, index=False, encoding="utf-8-sig")

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "coverage_threshold": COVERAGE_THRESHOLD,
        "stable_days": STABLE_DAYS,
        "cutoff_date": cutoff_date,
        "input_rows": input_rows,
        "input_days": input_days,
        "dropped_rows_warmup": dropped_rows,
        "dropped_days_warmup": dropped_days,
        "output_rows": int(len(df_kept)),
        "output_days": int(df_kept["TradeDate"].nunique()),
        "output_start_date": str(df_kept["TradeDate"].min()),
        "output_end_date": str(df_kept["TradeDate"].max()),
        "missing_before_all_cols_nonzero": missing_before_all,
        "step_results": step_stats,
        "missing_after_all_cols_nonzero": remaining_missing,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
