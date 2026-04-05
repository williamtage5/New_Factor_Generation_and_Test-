from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from factor_config import Config
from factor_metrics import (
    compute_group_returns_daily,
    compute_ic_series,
    summarize_distribution,
    summarize_group_returns,
    summarize_ic_decay,
)
from factor_plots import plot_factor_hist, plot_group_heatmap, plot_ic_decay


def run_for_factor(
    panel: pd.DataFrame, factor_col: str, cfg: Config
) -> dict[str, float | int | str]:
    factor_dir = cfg.output_dir / factor_col
    factor_dir.mkdir(parents=True, exist_ok=True)

    ic_df = compute_ic_series(
        panel=panel,
        factor_col=factor_col,
        ic_lags=cfg.ic_lags,
        min_ic_samples=cfg.min_ic_samples,
    )
    ic_decay_df = summarize_ic_decay(ic_df, cfg.ic_lags)
    grp_daily = compute_group_returns_daily(
        panel=panel,
        factor_col=factor_col,
        n_groups=cfg.n_groups,
        min_group_samples=cfg.min_group_samples,
    )
    grp_stats = summarize_group_returns(grp_daily)
    dist_stats = summarize_distribution(panel, factor_col)

    ic_df["TradeDate"] = pd.to_datetime(ic_df["TradeDate"]).dt.strftime("%Y-%m-%d")
    ic_df.to_csv(factor_dir / "ic_series.csv", index=False, encoding="utf-8-sig")
    ic_decay_df.to_csv(factor_dir / "ic_decay.csv", index=False, encoding="utf-8-sig")
    grp_daily.to_csv(factor_dir / "group_returns_daily.csv", index=False, encoding="utf-8-sig")
    grp_stats.to_csv(factor_dir / "group_returns_stats.csv", index=False, encoding="utf-8-sig")
    dist_stats.to_csv(
        factor_dir / "factor_distribution_stats.csv", index=False, encoding="utf-8-sig"
    )

    plot_ic_decay(ic_decay_df, factor_dir / "ic_decay.png", factor_col)
    plot_factor_hist(panel, factor_col, factor_dir / "factor_hist.png", cfg.hist_sample_size)
    plot_group_heatmap(grp_daily, factor_dir / "group_heatmap.png", cfg.n_groups, factor_col)

    lag1 = ic_decay_df.loc[ic_decay_df["lag"] == 1].head(1)
    lag1_mean_ic = float(lag1["mean_ic"].iloc[0]) if not lag1.empty else np.nan
    lag1_icir = float(lag1["icir_annualized"].iloc[0]) if not lag1.empty else np.nan

    ls_row = grp_stats.loc[grp_stats["group"] == "LS_Gn_G1"].head(1)
    ls_mean = float(ls_row["mean_daily"].iloc[0]) if not ls_row.empty else np.nan
    ls_ir = float(ls_row["ir"].iloc[0]) if not ls_row.empty else np.nan
    ls_cum = float(ls_row["cum_return"].iloc[0]) if not ls_row.empty else np.nan

    return {
        "factor": factor_col,
        "ic_days_lag1": int(lag1["count"].iloc[0]) if not lag1.empty else 0,
        "ic_mean_lag1": lag1_mean_ic,
        "icir_lag1_annualized": lag1_icir,
        "group_days": int(len(grp_daily)),
        "ls_mean_daily": ls_mean,
        "ls_ir": ls_ir,
        "ls_cum_return": ls_cum,
        "output_dir": str(factor_dir),
    }


def write_summary(
    summary_records: list[dict[str, float | int | str]], output_dir: Path
) -> None:
    summary_df = pd.DataFrame(summary_records).sort_values("factor").reset_index(drop=True)
    summary_df.to_csv(output_dir / "all_factors_summary.csv", index=False, encoding="utf-8-sig")


def write_manifest(
    cfg: Config, factor_cols: list[str], panel: pd.DataFrame, output_dir: Path
) -> None:
    manifest = {
        "input_dir": str(cfg.input_dir),
        "output_dir": str(cfg.output_dir),
        "factor_suffix": cfg.factor_suffix,
        "factor_count": len(factor_cols),
        "factors": factor_cols,
        "ic_lags": cfg.ic_lags,
        "n_groups": cfg.n_groups,
        "min_ic_samples": cfg.min_ic_samples,
        "min_group_samples": cfg.min_group_samples,
        "rows_total": int(len(panel)),
        "days_total": int(panel["TradeDate"].nunique()),
        "stocks_per_day_min": int(panel.groupby("TradeDate")["StockCode"].nunique().min()),
        "stocks_per_day_max": int(panel.groupby("TradeDate")["StockCode"].nunique().max()),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
