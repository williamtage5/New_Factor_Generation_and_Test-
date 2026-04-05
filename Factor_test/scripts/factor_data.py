from __future__ import annotations

from pathlib import Path

import pandas as pd


def detect_factor_columns(sample_file: Path, factor_suffix: str) -> list[str]:
    sample_cols = pd.read_csv(sample_file, nrows=0, encoding="utf-8-sig").columns
    factor_cols = [c for c in sample_cols if c.endswith(factor_suffix)]
    if not factor_cols:
        raise ValueError(
            f"No factor columns ending with '{factor_suffix}' in {sample_file}"
        )
    return factor_cols


def load_panel(input_dir: Path, factor_cols: list[str], ic_lags: list[int]) -> pd.DataFrame:
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found in {input_dir}")

    usecols = ["TradeDate", "StockCode", "ClosePrice"] + factor_cols
    panel = pd.concat(
        [
            pd.read_csv(f, usecols=usecols, encoding="utf-8-sig", low_memory=False)
            for f in files
        ],
        ignore_index=True,
    )

    panel["TradeDate"] = pd.to_datetime(panel["TradeDate"].astype(str), format="%Y%m%d")
    panel["ClosePrice"] = pd.to_numeric(panel["ClosePrice"], errors="coerce")
    for col in factor_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    panel = panel.sort_values(["StockCode", "TradeDate"]).reset_index(drop=True)
    for lag in ic_lags:
        panel[f"fwd_ret_{lag}d"] = (
            panel.groupby("StockCode", sort=False)["ClosePrice"].shift(-lag)
            / panel["ClosePrice"]
            - 1.0
        )

    panel = panel.sort_values(["TradeDate", "StockCode"]).reset_index(drop=True)
    return panel
