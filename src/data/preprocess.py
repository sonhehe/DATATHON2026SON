from __future__ import annotations

import pandas as pd


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    y = df[target_col]
    x = df.drop(columns=[target_col])
    return x, y


def cast_datetime_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    keep_cols = df.columns[df.isna().mean() <= threshold]
    return df[keep_cols].copy()
