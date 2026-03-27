from __future__ import annotations

import numpy as np
import pandas as pd


def add_zero_flags(df: pd.DataFrame, numeric_cols: list[str] | None = None, min_zero_ratio: float = 0.7) -> pd.DataFrame:
    out = df.copy()
    if numeric_cols is None:
        numeric_cols = out.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:
        ratio = (out[col] == 0).mean() if col in out.columns else 0
        if ratio >= min_zero_ratio:
            out[f"is_zero__{col}"] = (out[col] == 0).astype(int)
    return out


def add_datetime_parts(df: pd.DataFrame, datetime_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in datetime_cols:
        if col not in out.columns:
            continue
        s = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}__year"] = s.dt.year
        out[f"{col}__month"] = s.dt.month
        out[f"{col}__day"] = s.dt.day
        out[f"{col}__dow"] = s.dt.dayofweek
    return out
