
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Load the cleaned dataset
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(
    ROOT / "data" / "Telco-Customer-Churn-Cleaned.csv"
)

# 2. Separate features from the target
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Identify feature types
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude="number"
).columns.tolist()

# 4. Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 5. Create the complete ML pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# 6. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 7. Train the model
model.fit(X_train, y_train)

# 8. Generate predictions
predictions = model.predict(X_test)

# 9. Evaluate performance
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(precision_score(y_test, predictions, zero_division=0)),
    "recall": float(recall_score(y_test, predictions, zero_division=0)),
    "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
}

print("\nMODEL RESULTS")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    target_names=["Stayed", "Churned"]
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

# 10. Save the model and metrics
output = ROOT / "artifacts"
output.mkdir(exist_ok=True)

joblib.dump(model, output / "customer_churn_model.joblib")
(output / "metrics.json").write_text(
    json.dumps(metrics, indent=2)
)

print("\nSaved model and metrics in artifacts/")
  