from __future__ import annotations

import numpy as np
import pandas as pd


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=constant_cols), constant_cols


def drop_high_cardinality_categoricals(df: pd.DataFrame, threshold: int = 200) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    drop_cols = [c for c in cat_cols if df[c].nunique(dropna=False) > threshold]
    return df.drop(columns=drop_cols), drop_cols


def correlation_ranking(df: pd.DataFrame, target: pd.Series, top_k: int = 30) -> pd.DataFrame:
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corrwith(target).abs().sort_values(ascending=False).head(top_k)
    return corr.rename("abs_corr_with_target").reset_index(names="feature")
