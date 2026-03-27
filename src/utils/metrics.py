from __future__ import annotations

import pandas as pd


def target_rate_by_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)[target_col]
        .agg(["count", "mean"])
        .rename(columns={"mean": "target_rate"})
        .sort_values("target_rate", ascending=False)
        .reset_index()
    )
    return out
