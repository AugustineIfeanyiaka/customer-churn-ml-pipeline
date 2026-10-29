
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(
    PROJECT_ROOT / "artifacts" / "optimized_churn_model.joblib"
)

customer = pd.DataFrame([{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
}])

customer = customer[model.feature_names_in_]

probability = model.predict_proba(customer)[0, 1]

print(f"Churn probability: {probability:.6f}")
print(f"Classification: {'High Risk' if probability >= 0.30 else 'Low Risk'}")
