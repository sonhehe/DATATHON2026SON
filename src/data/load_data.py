from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_EXTENSIONS = {".csv", ".parquet"}


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular file from CSV or Parquet."""
    path = Path(path)
    if path.suffix not in REQUIRED_EXTENSIONS:
        raise ValueError(f"Unsupported extension {path.suffix}. Use one of {REQUIRED_EXTENSIONS}.")

    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def load_train_test(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_table(train_path)
    test_df = load_table(test_path)
    return train_df, test_df
