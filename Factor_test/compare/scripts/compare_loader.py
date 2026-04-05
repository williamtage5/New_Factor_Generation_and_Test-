from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "ic_decay.csv",
    "ic_series.csv",
    "group_returns_stats.csv",
    "group_returns_daily.csv",
    "factor_distribution_stats.csv",
]


def discover_factor_dirs(metrics_input_dir: Path) -> list[Path]:
    if not metrics_input_dir.exists():
        raise FileNotFoundError(f"Metrics input directory not found: {metrics_input_dir}")

    factor_dirs: list[Path] = []
    for path in sorted(metrics_input_dir.iterdir()):
        if not path.is_dir():
            continue
        if all((path / filename).exists() for filename in REQUIRED_FILES):
            factor_dirs.append(path)
    if not factor_dirs:
        raise ValueError(f"No valid factor directories found under: {metrics_input_dir}")
    return factor_dirs


def load_factor_data(factor_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "ic_decay": pd.read_csv(factor_dir / "ic_decay.csv"),
        "ic_series": pd.read_csv(factor_dir / "ic_series.csv"),
        "group_returns_stats": pd.read_csv(factor_dir / "group_returns_stats.csv"),
        "group_returns_daily": pd.read_csv(factor_dir / "group_returns_daily.csv"),
        "factor_distribution_stats": pd.read_csv(
            factor_dir / "factor_distribution_stats.csv"
        ),
    }
