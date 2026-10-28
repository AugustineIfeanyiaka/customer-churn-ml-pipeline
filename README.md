# Customer Churn Prediction — Machine Learning Pipeline

## Project Overview

An end-to-end machine learning project predicting customer churn using the IBM Telco Customer Churn sample dataset.

The project demonstrates data cleaning, exploratory analysis, preprocessing, classification, cross-validation, threshold optimization, and model evaluation using Python and scikit-learn.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Logistic Regression, Random Forest, Git and GitHub.

## Dataset

- Source: IBM Telco Customer Churn sample dataset
- Original records: 7,043
- Cleaned records: 7,032
- Features include tenure, contract type, monthly charges, internet service and payment method.
- Target: `Churn` (Yes/No)

The dataset contains fictional customer records intended for educational analysis.

## Methodology

1. Converted `TotalCharges` to numeric and investigated 11 missing values.
2. Removed the 11 affected records for this baseline experiment.
3. Excluded customer identifiers from model features.
4. Used an 80/20 stratified training and test split.
5. Built leakage-safe preprocessing pipelines for numerical and categorical features.
6. Compared Logistic Regression and Random Forest using five-fold cross-validation.
7. Selected a classification threshold using out-of-fold training predictions.
8. Evaluated the selected threshold on a held-out test set.

## Results

### Five-fold cross-validation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.25% | 78.49% |
| Precision | 65.40% | 62.11% |
| Recall | 54.78% | 48.90% |
| F1-score | 59.60% | 54.71% |

Logistic Regression performed better across all four reported metrics.

### Held-out test evaluation

| Metric | Threshold 0.50 | Threshold 0.30 |
|---|---:|---:|
| Accuracy | 80.38% | 74.27% |
| Precision | 64.85% | 51.08% |
| Recall | 57.22% | 75.94% |
| F1-score | 60.80% | 61.08% |
| True positives | 214 | 284 |
| False positives | 116 | 272 |
| False negatives | 160 | 90 |

**Key finding:** Lowering the classification threshold to 0.30 increased churn recall by 18.72 percentage points and detected 70 additional churners on the held-out test set.

This improvement increased false-positive alerts, demonstrating the business tradeoff between recall and precision.

## Running the Project

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the IBM Telco Customer Churn CSV into `data/Telco-Customer-Churn.csv`.

```bash
python src/explore_data.py
python src/train_churn.py
python src/compare_models.py
python src/threshold_tuning.py
```

Trained models and selected thresholds are saved locally in `artifacts/`.

## Limitations and Future Improvements

- Evaluate other classifiers and hyperparameters.
- Investigate the effect of excluding customers with zero tenure.
- Add prediction-serving functionality using FastAPI.
- Add automated tests and CI/CD.
- Explore cost-sensitive decision thresholds for retention campaigns.
- Assess model performance and stability on additional datasets.

## Author

Augustine Ifeanyiaka

Python | Machine Learning | AI Engineering

This is an educational portfolio project, not a deployed telecommunications production system.