from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid")


def plot_bar_icir(summary: pd.DataFrame, out_file: Path) -> None:
    df = summary.sort_values("icir_1d", ascending=True)
    colors = ["#2E7D32" if x >= 0 else "#C62828" for x in df["icir_1d"]]

    plt.figure(figsize=(10, max(6, 0.35 * len(df))))
    plt.barh(df["factor"], df["icir_1d"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Factor ICIR (1d)")
    plt.xlabel("ICIR")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def plot_bar_ls_ir(summary: pd.DataFrame, out_file: Path) -> None:
    df = summary.sort_values("ls_ir", ascending=True)
    colors = ["#1565C0" if x >= 0 else "#EF6C00" for x in df["ls_ir"]]

    plt.figure(figsize=(10, max(6, 0.35 * len(df))))
    plt.barh(df["factor"], df["ls_ir"], color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Factor Long-Short IR (G10-G1)")
    plt.xlabel("Long-Short IR")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def plot_heatmap_ic_lag(ic_lag_matrix: pd.DataFrame, out_file: Path) -> None:
    if ic_lag_matrix.empty:
        return

    heat = ic_lag_matrix.set_index("factor")
    width = max(8, 1.3 * len(heat.columns))
    height = max(6, 0.4 * len(heat.index))
    plt.figure(figsize=(width, height))
    sns.heatmap(heat, cmap="RdBu_r", center=0.0, annot=False)
    plt.title("IC Mean Heatmap by Lag")
    plt.xlabel("Lag")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def plot_heatmap_ls_yearly(ls_yearly_matrix: pd.DataFrame, out_file: Path) -> None:
    if ls_yearly_matrix.empty:
        return

    heat = ls_yearly_matrix.set_index("factor")
    width = max(10, 1.0 * len(heat.columns))
    height = max(6, 0.4 * len(heat.index))
    plt.figure(figsize=(width, height))
    sns.heatmap(heat, cmap="RdYlGn", center=0.0, annot=False)
    plt.title("Long-Short Mean Return Heatmap by Year")
    plt.xlabel("Year")
    plt.ylabel("Factor")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def plot_scatter_ic_vs_ls(summary: pd.DataFrame, out_file: Path) -> None:
    df = summary.copy()
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="icir_1d",
        y="ls_ir",
        hue="ic_direction_1d",
        s=90,
        palette={"positive": "#2E7D32", "negative": "#C62828", "flat": "#616161"},
    )
    for _, r in df.iterrows():
        plt.text(r["icir_1d"], r["ls_ir"], r["factor"], fontsize=8, alpha=0.9)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("ICIR(1d) vs Long-Short IR")
    plt.xlabel("ICIR(1d)")
    plt.ylabel("Long-Short IR")
    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()


def plot_rank_top_bottom(summary: pd.DataFrame, out_file: Path, top_n: int) -> None:
    rank = summary.sort_values("score_total", ascending=False).reset_index(drop=True)
    top = rank.head(top_n).copy()
    bottom = rank.tail(top_n).copy().sort_values("score_total", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.5 * top_n)))
    axes[0].barh(top["factor"], top["score_total"], color="#1B5E20")
    axes[0].set_title(f"Top {top_n} Composite Score")
    axes[0].set_xlabel("Score")
    axes[0].invert_yaxis()

    axes[1].barh(bottom["factor"], bottom["score_total"], color="#B71C1C")
    axes[1].set_title(f"Bottom {top_n} Composite Score")
    axes[1].set_xlabel("Score")
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_file, dpi=170)
    plt.close()
