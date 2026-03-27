from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.evaluate import ClassificationResult, evaluate_classification
from src.utils.config import DEFAULT_RANDOM_STATE


@dataclass
class BaselineArtifacts:
    pipeline: Pipeline
    result: ClassificationResult


def train_baseline_classifier(df: pd.DataFrame, target_col: str) -> BaselineArtifacts:
    x = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = x.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])

    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, random_state=DEFAULT_RANDOM_STATE, stratify=y
    )

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_valid)
    y_proba = pipe.predict_proba(x_valid)[:, 1] if hasattr(pipe[-1], "predict_proba") else None
    result = evaluate_classification(y_valid, y_pred, y_proba)

    return BaselineArtifacts(pipeline=pipe, result=result)
