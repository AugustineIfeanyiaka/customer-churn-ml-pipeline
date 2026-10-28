from pathlib import Path
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_predict,
    StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "data" / "Telco-Customer-Churn-Cleaned.csv"
)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

numeric = X.select_dtypes(
    include="number"
).columns.tolist()

categorical = X.select_dtypes(
    exclude="number"
).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Out-of-fold probabilities from training data only
probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

print("Threshold  Precision  Recall  F1")

results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_train, predictions, zero_division=0
    )
    recall = recall_score(
        y_train, predictions, zero_division=0
    )
    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    results.append((threshold, f1))

    print(
        f"{threshold:.2f}       "
        f"{precision:.4f}     "
        f"{recall:.4f}  "
        f"{f1:.4f}"
    )

best_threshold = max(
    results,
    key=lambda result: result[1]
)[0]

print("\nBest threshold by F1:", best_threshold)

# FINAL EVALUATION ON HELD-OUT TEST DATA

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import joblib
import json

# Train on the full training dataset
model.fit(X_train, y_train)

# Predict churn probabilities on unseen test data
test_probabilities = model.predict_proba(X_test)[:, 1]

# Compare default and selected threshold
for threshold in [0.50, best_threshold]:

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    print(f"\nTEST RESULTS — Threshold {threshold:.2f}")

    print("Accuracy:",
          round(accuracy_score(y_test, test_predictions), 4))

    print("Precision:",
          round(precision_score(y_test, test_predictions,
                                zero_division=0), 4))

    print("Recall:",
          round(recall_score(y_test, test_predictions,
                             zero_division=0), 4))

    print("F1-score:",
          round(f1_score(y_test, test_predictions,
                         zero_division=0), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, test_predictions,
        target_names=["Stayed", "Churned"]
    ))

# Save final optimized model and its threshold
output = ROOT / "artifacts"
output.mkdir(exist_ok=True)

joblib.dump(
    model,
    output / "optimized_churn_model.joblib"
)

(output / "optimized_threshold.json").write_text(
    json.dumps(
        {"classification_threshold": float(best_threshold)},
        indent=2
    )
)

print("\nOptimized model and threshold saved!")
