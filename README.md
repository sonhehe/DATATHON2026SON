# Datathon Analysis Starter Kit

Bộ khung này tập trung vào **phân tích dữ liệu mạnh trước modeling**, phù hợp workflow datathon.

## Cấu trúc dự án

```text
datathon_project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_ideas.ipynb
│   └── 04_baseline_model.ipynb
│
├── src/
│   ├── data/
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   ├── build_features.py
│   │   └── feature_selection.py
│   ├── models/
│   │   ├── train_baseline.py
│   │   ├── evaluate.py
│   │   └── explain.py
│   └── utils/
│       ├── config.py
│       ├── plots.py
│       └── metrics.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
│
├── requirements.txt
├── rule.md
└── main.py
```

## Cách dùng nhanh

1. Cập nhật `rule.md` theo đề bài datathon.
2. Đặt dữ liệu vào `data/raw/` (mặc định: `train.csv`, `test.csv`).
3. Chạy audit:

```bash
python main.py audit --train data/raw/train.csv --test data/raw/test.csv --target target
```

4. Chạy baseline:

```bash
python main.py baseline --train data/raw/train.csv --target target
```

5. Mở notebooks để phân tích sâu theo thứ tự 01 -> 04.

## Lưu ý

- `main.py` ưu tiên reproducibility và kiểm tra leakage cơ bản.
- Các notebooks đã có template cho univariate/bivariate/multivariate.
- Output report/tables sẽ nằm trong `outputs/`.
