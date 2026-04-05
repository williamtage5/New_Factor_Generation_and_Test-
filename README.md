# New_Factor_Generation_and_Test-

Factor generation and single-factor testing pipeline for CSI 300 constituents.

## Overview

This project includes:

- Data request and preprocessing
- Factor feature generation
- Single-factor IC analysis
- Decile group backtesting (10 groups)
- Factor report/plot output

## Repository Structure

- `Data_Request/`
- `Data_Request/sql_request/`: SQL templates for raw data pulls
- `Data_Request/data_processing/`: preprocessing and factor generation scripts
- `Data_Request/data/`: raw/merged/cleaned datasets
- `Factor_test/`
- `Factor_test/scripts/`: single-factor test pipeline scripts
- `Factor_test/compare/scripts/`: cross-factor comparison scripts
- `Factor_test/data/metrics/`: per-factor IC/group stats and figures
- `结果分析报告.md`: generated result analysis report

## Main Scripts

- `Data_Request/data_processing/build_daily_merged_csv.py`
- `Data_Request/data_processing/handle_missing_values.py`
- `Data_Request/data_processing/handle_outliers_3sigma.py`
- `Data_Request/data_processing/generate_factor_features.py`
- `Factor_test/scripts/run_all_factor_metrics.py`
- `Factor_test/compare/scripts/build_compare_report.py`

## Typical Workflow

1. Prepare and merge daily data in `Data_Request/data/`.
2. Run missing-value and outlier handling.
3. Generate factors in `Data_Request/data/factor_generation/`.
4. Run single-factor metrics and decile backtests.
5. Review outputs in `Factor_test/data/metrics/` and `结果分析报告.md`.

## Output Artifacts

For each factor under `Factor_test/data/metrics/<factor_name>/`:

- `ic_series.csv`: daily IC time series
- `ic_decay.csv`: IC decay across lags
- `group_returns_stats.csv`: decile and long-short summary table
- `factor_distribution_stats.csv`: distribution summary
- `ic_decay.png`: IC decay curve
- `factor_hist.png`: factor histogram
- `group_heatmap.png`: decile return heatmap
