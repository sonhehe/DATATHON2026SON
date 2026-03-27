from __future__ import annotations

import argparse
from pathlib import Path

from src.data.load_data import load_table, load_train_test
from src.data.validate_data import basic_audit, build_schema_table, detect_id_like_columns, detect_numeric_outliers_iqr
from src.models.train_baseline import train_baseline_classifier
from src.utils.config import Paths


def run_audit(train_path: str, test_path: str, target_col: str) -> None:
    paths = Paths()
    paths.outputs_tables.mkdir(parents=True, exist_ok=True)
    paths.outputs_reports.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test(train_path, test_path)
    schema = build_schema_table(train_df)
    schema.to_csv(paths.outputs_tables / "train_schema.csv", index=False)

    outliers = detect_numeric_outliers_iqr(train_df, exclude=[target_col])
    outliers.to_csv(paths.outputs_tables / "train_outliers_iqr.csv", index=False)

    audit = basic_audit(train_df, target_col)
    id_like = detect_id_like_columns(train_df.drop(columns=[target_col], errors="ignore"))

    report = f"""# Data Audit Report

## Shapes
- Train: {train_df.shape}
- Test: {test_df.shape}

## Core checks
- Duplicated rows (train): {audit.duplicated_rows}
- Target missing ratio: {audit.target_missing_ratio:.4f}
- ID/quasi-ID suspects: {', '.join(id_like) if id_like else 'None'}

## Exported artifacts
- outputs/tables/train_schema.csv
- outputs/tables/train_outliers_iqr.csv
"""
    (paths.outputs_reports / "data_audit.md").write_text(report, encoding="utf-8")
    print("[OK] Audit completed. See outputs/reports/data_audit.md")


def run_baseline(train_path: str, target_col: str) -> None:
    paths = Paths()
    paths.outputs_reports.mkdir(parents=True, exist_ok=True)

    train_df = load_table(train_path)
    artifacts = train_baseline_classifier(train_df, target_col)
    r = artifacts.result

    report = f"""# Baseline Report

## Logistic Regression baseline
- Accuracy: {r.accuracy:.4f}
- F1: {r.f1:.4f}
- ROC AUC: {r.roc_auc if r.roc_auc is not None else 'N/A'}
"""
    (paths.outputs_reports / "baseline_report.md").write_text(report, encoding="utf-8")
    print("[OK] Baseline completed. See outputs/reports/baseline_report.md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Datathon analysis toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Run data audit")
    audit.add_argument("--train", required=True)
    audit.add_argument("--test", required=True)
    audit.add_argument("--target", required=True)

    baseline = sub.add_parser("baseline", help="Train baseline model")
    baseline.add_argument("--train", required=True)
    baseline.add_argument("--target", required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        run_audit(args.train, args.test, args.target)
    elif args.command == "baseline":
        run_baseline(args.train, args.target)


if __name__ == "__main__":
    main()
