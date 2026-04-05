from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    input_dir: Path
    output_dir: Path
    factor_suffix: str
    ic_lags: list[int]
    n_groups: int
    min_ic_samples: int
    min_group_samples: int
    hist_sample_size: int


def parse_args() -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "Data_Request" / "data" / "handled_outliers"
    default_output = repo_root / "Factor_test" / "data" / "metrics"

    parser = argparse.ArgumentParser(
        description="Run IC and decile group tests for all factor columns."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--factor-suffix", type=str, default="_clip3")
    parser.add_argument("--ic-lags", type=int, nargs="+", default=[1, 5, 10, 20])
    parser.add_argument("--n-groups", type=int, default=10)
    parser.add_argument("--min-ic-samples", type=int, default=30)
    parser.add_argument("--min-group-samples", type=int, default=100)
    parser.add_argument("--hist-sample-size", type=int, default=200_000)
    args = parser.parse_args()

    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        factor_suffix=args.factor_suffix,
        ic_lags=sorted(set(args.ic_lags)),
        n_groups=args.n_groups,
        min_ic_samples=args.min_ic_samples,
        min_group_samples=args.min_group_samples,
        hist_sample_size=args.hist_sample_size,
    )
