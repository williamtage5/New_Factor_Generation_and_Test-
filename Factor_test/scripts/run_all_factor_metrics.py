from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from factor_config import parse_args
from factor_data import detect_factor_columns, load_panel
from factor_pipeline import run_for_factor, write_manifest, write_summary
from factor_plots import set_plot_theme


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cfg.input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No csv files found in {cfg.input_dir}")

    factor_cols = detect_factor_columns(files[0], cfg.factor_suffix)
    panel = load_panel(cfg.input_dir, factor_cols=factor_cols, ic_lags=cfg.ic_lags)

    summary_records: list[dict[str, float | int | str]] = []
    for i, factor in enumerate(factor_cols, start=1):
        print(f"[{i}/{len(factor_cols)}] processing {factor}")
        summary_records.append(run_for_factor(panel, factor_col=factor, cfg=cfg))

    write_summary(summary_records=summary_records, output_dir=cfg.output_dir)
    write_manifest(cfg=cfg, factor_cols=factor_cols, panel=panel, output_dir=cfg.output_dir)
    print("Done.")


if __name__ == "__main__":
    set_plot_theme()
    main()
