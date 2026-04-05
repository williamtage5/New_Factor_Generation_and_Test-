from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def set_plot_theme() -> None:
    sns.set_theme(style="whitegrid")


def plot_ic_decay(ic_decay_df: pd.DataFrame, out_file: Path, factor_name: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(ic_decay_df["lag"], ic_decay_df["mean_ic"], marker="o", linewidth=2)
    plt.axhline(0, color="black", linewidth=1, alpha=0.7)
    plt.title(f"{factor_name} IC Decay")
    plt.xlabel("Lag (days)")
    plt.ylabel("Mean IC")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def plot_factor_hist(
    panel: pd.DataFrame, factor_col: str, out_file: Path, sample_size: int
) -> None:
    series = pd.to_numeric(panel[factor_col], errors="coerce").dropna()
    if len(series) > sample_size:
        series = series.sample(sample_size, random_state=42)

    plt.figure(figsize=(8, 5))
    sns.histplot(series, bins=80, kde=False)
    plt.title(f"{factor_col} Distribution")
    plt.xlabel("Factor Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()


def plot_group_heatmap(
    group_daily: pd.DataFrame, out_file: Path, n_groups: int, factor_name: str
) -> None:
    if group_daily.empty:
        return

    tmp = group_daily.copy()
    tmp["TradeDate"] = pd.to_datetime(tmp["TradeDate"])
    group_cols = [f"G{i}" for i in range(1, n_groups + 1)]
    long_df = tmp.melt(
        id_vars=["TradeDate"],
        value_vars=group_cols,
        var_name="Group",
        value_name="Return",
    )
    long_df["Year"] = long_df["TradeDate"].dt.year
    pivot = long_df.pivot_table(
        index="Group", columns="Year", values="Return", aggfunc="mean"
    )
    pivot = pivot.reindex(group_cols)

    width = max(8, 1.2 * len(pivot.columns))
    plt.figure(figsize=(width, 5.5))
    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0.0,
        annot=False,
        cbar_kws={"label": "Average Daily Return"},
    )
    plt.title(f"{factor_name} Group Return Heatmap (Yearly Avg)")
    plt.xlabel("Year")
    plt.ylabel("Group")
    plt.tight_layout()
    plt.savefig(out_file, dpi=160)
    plt.close()
