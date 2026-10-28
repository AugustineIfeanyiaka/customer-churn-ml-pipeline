"""Leakage-safe baseline for the IBM Telco Customer Churn example dataset."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
OUTPUT = ROOT / "outputs"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    required = {"Churn", "TotalCharges", "customerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data["Churn"].map({"Yes": 1, "No": 0})
    if target.isna().any():
        raise ValueError("Churn must only contain 'Yes' or 'No'")
    features = data.drop(columns=["Churn", "customerID"])
    return features, target.astype(int)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, numeric_columns),
        ("categorical", categorical, categorical_columns),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\nDownload the IBM Telco CSV as explained in README.md."
        )
    X, y = prepare_data(pd.read_csv(DATA))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(pipeline, OUTPUT / "churn_pipeline.joblib")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
