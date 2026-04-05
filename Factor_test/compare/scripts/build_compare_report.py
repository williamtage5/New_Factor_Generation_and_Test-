from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_config import parse_args
from compare_loader import discover_factor_dirs, load_factor_data
from compare_metrics import (
    build_master_summary,
    build_quality_flags,
    build_rank_tables,
)
from compare_plots import (
    plot_bar_icir,
    plot_bar_ls_ir,
    plot_heatmap_ic_lag,
    plot_heatmap_ls_yearly,
    plot_rank_top_bottom,
    plot_scatter_ic_vs_ls,
    set_plot_theme,
)


def main() -> None:
    cfg = parse_args()
    cfg.output_data_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(parents=True, exist_ok=True)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    factor_dirs = discover_factor_dirs(cfg.metrics_input_dir)
    factor_data: dict[str, dict[str, pd.DataFrame]] = {}
    for i, factor_dir in enumerate(factor_dirs, start=1):
        factor_name = factor_dir.name
        print(f"[{i}/{len(factor_dirs)}] loading {factor_name}")
        factor_data[factor_name] = load_factor_data(factor_dir)

    summary, ic_lag_matrix, ls_yearly_matrix = build_master_summary(
        factor_data=factor_data,
        ic_lags=cfg.ic_lags,
    )
    rank_ic, rank_ls, rank_composite = build_rank_tables(summary)
    quality_flags = build_quality_flags(
        summary=summary,
        icir_abs_threshold=cfg.icir_abs_threshold,
        ls_ir_abs_threshold=cfg.ls_ir_abs_threshold,
        ls_win_rate_threshold=cfg.ls_win_rate_threshold,
    )

    summary.to_csv(cfg.tables_dir / "factor_master_summary.csv", index=False, encoding="utf-8-sig")
    rank_ic.to_csv(cfg.tables_dir / "factor_rank_ic.csv", index=False, encoding="utf-8-sig")
    rank_ls.to_csv(cfg.tables_dir / "factor_rank_ls.csv", index=False, encoding="utf-8-sig")
    rank_composite.to_csv(
        cfg.tables_dir / "factor_rank_composite.csv", index=False, encoding="utf-8-sig"
    )
    ic_lag_matrix.to_csv(cfg.tables_dir / "ic_lag_matrix.csv", index=False, encoding="utf-8-sig")
    ls_yearly_matrix.to_csv(
        cfg.tables_dir / "ls_yearly_matrix.csv", index=False, encoding="utf-8-sig"
    )
    quality_flags.to_csv(cfg.tables_dir / "quality_flags.csv", index=False, encoding="utf-8-sig")

    set_plot_theme()
    plot_bar_icir(summary, cfg.figures_dir / "bar_icir.png")
    plot_bar_ls_ir(summary, cfg.figures_dir / "bar_ls_ir.png")
    plot_heatmap_ic_lag(ic_lag_matrix, cfg.figures_dir / "heatmap_ic_lag.png")
    plot_heatmap_ls_yearly(ls_yearly_matrix, cfg.figures_dir / "heatmap_ls_yearly.png")
    plot_scatter_ic_vs_ls(summary, cfg.figures_dir / "scatter_ic_vs_ls.png")
    plot_rank_top_bottom(summary, cfg.figures_dir / "rank_top_bottom.png", top_n=cfg.top_n)

    manifest = {
        "metrics_input_dir": str(cfg.metrics_input_dir),
        "output_data_dir": str(cfg.output_data_dir),
        "factor_count": int(len(summary)),
        "factors": sorted(summary["factor"].tolist()),
        "ic_lags": cfg.ic_lags,
        "thresholds": {
            "icir_abs_threshold": cfg.icir_abs_threshold,
            "ls_ir_abs_threshold": cfg.ls_ir_abs_threshold,
            "ls_win_rate_threshold": cfg.ls_win_rate_threshold,
        },
        "generated_tables": [
            "factor_master_summary.csv",
            "factor_rank_ic.csv",
            "factor_rank_ls.csv",
            "factor_rank_composite.csv",
            "ic_lag_matrix.csv",
            "ls_yearly_matrix.csv",
            "quality_flags.csv",
        ],
        "generated_figures": [
            "bar_icir.png",
            "bar_ls_ir.png",
            "heatmap_ic_lag.png",
            "heatmap_ls_yearly.png",
            "scatter_ic_vs_ls.png",
            "rank_top_bottom.png",
        ],
    }
    (cfg.output_data_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Done.")


if __name__ == "__main__":
    main()
