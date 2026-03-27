from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AuditSummary:
    n_rows: int
    n_cols: int
    duplicated_rows: int
    target_missing_ratio: float


def build_schema_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "missing_count": df.isna().sum().values,
            "missing_ratio": df.isna().mean().values,
            "n_unique": df.nunique(dropna=False).values,
            "unique_ratio": (df.nunique(dropna=False) / len(df)).values,
        }
    )
    return out.sort_values(["missing_ratio", "n_unique"], ascending=[False, False]).reset_index(drop=True)


def detect_id_like_columns(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    nunique_ratio = df.nunique(dropna=False) / len(df)
    return nunique_ratio[nunique_ratio >= threshold].index.tolist()


def detect_numeric_outliers_iqr(df: pd.DataFrame, exclude: list[str] | None = None) -> pd.DataFrame:
    exclude = exclude or []
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]
    rows: list[dict[str, float | str]] = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            rows.append({"column": col, "outlier_ratio": 0.0})
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            rows.append({"column": col, "outlier_ratio": 0.0})
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ratio = ((s < lower) | (s > upper)).mean()
        rows.append({"column": col, "outlier_ratio": float(ratio)})

    return pd.DataFrame(rows).sort_values("outlier_ratio", ascending=False).reset_index(drop=True)


def basic_audit(df: pd.DataFrame, target_col: str) -> AuditSummary:
    target_missing = float(df[target_col].isna().mean()) if target_col in df.columns else 1.0
    return AuditSummary(
        n_rows=len(df),
        n_cols=df.shape[1],
        duplicated_rows=int(df.duplicated().sum()),
        target_missing_ratio=target_missing,
    )
