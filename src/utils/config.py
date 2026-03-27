from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(".")
    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    outputs_tables: Path = Path("outputs/tables")
    outputs_reports: Path = Path("outputs/reports")
    outputs_figures: Path = Path("outputs/figures")


DEFAULT_RANDOM_STATE = 42
