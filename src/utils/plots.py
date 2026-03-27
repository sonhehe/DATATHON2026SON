from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_histogram(series, output_path: str | Path, bins: int = 30, title: str | None = None) -> None:
    plt.figure(figsize=(8, 4))
    sns.histplot(series.dropna(), bins=bins)
    plt.title(title or f"Distribution of {series.name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def save_boxplot_by_target(df, numeric_col: str, target_col: str, output_path: str | Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=target_col, y=numeric_col)
    plt.title(f"{numeric_col} by {target_col}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
