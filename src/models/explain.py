from __future__ import annotations

import pandas as pd


def summarize_feature_hypotheses() -> pd.DataFrame:
    """Template bảng để ghi ý tưởng feature và bằng chứng EDA."""
    return pd.DataFrame(
        columns=[
            "feature_name",
            "feature_type",
            "hypothesis",
            "evidence",
            "leakage_risk",
            "priority",
            "owner",
            "status",
        ]
    )
