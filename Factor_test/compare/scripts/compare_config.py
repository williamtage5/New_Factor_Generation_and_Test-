from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompareConfig:
    metrics_input_dir: Path
    output_data_dir: Path
    tables_dir: Path
    figures_dir: Path
    ic_lags: list[int]
    icir_abs_threshold: float
    ls_ir_abs_threshold: float
    ls_win_rate_threshold: float
    top_n: int


def parse_args() -> CompareConfig:
    factor_test_dir = Path(__file__).resolve().parents[2]
    default_metrics_input = factor_test_dir / "data" / "metrics"
    default_output_data = factor_test_dir / "compare" / "data"

    parser = argparse.ArgumentParser(
        description="Build cross-factor comparison tables and figures."
    )
    parser.add_argument("--metrics-input-dir", type=Path, default=default_metrics_input)
    parser.add_argument("--output-data-dir", type=Path, default=default_output_data)
    parser.add_argument("--ic-lags", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--icir-abs-threshold", type=float, default=0.5)
    parser.add_argument("--ls-ir-abs-threshold", type=float, default=0.2)
    parser.add_argument("--ls-win-rate-threshold", type=float, default=0.52)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    output_data_dir = args.output_data_dir
    tables_dir = output_data_dir / "tables"
    figures_dir = output_data_dir / "figures"

    return CompareConfig(
        metrics_input_dir=args.metrics_input_dir,
        output_data_dir=output_data_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        ic_lags=sorted(set(args.ic_lags)),
        icir_abs_threshold=args.icir_abs_threshold,
        ls_ir_abs_threshold=args.ls_ir_abs_threshold,
        ls_win_rate_threshold=args.ls_win_rate_threshold,
        top_n=args.top_n,
    )
