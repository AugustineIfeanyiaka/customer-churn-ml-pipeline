# Customer Churn Prediction — Machine Learning Pipeline

**Status:** Starter project; no results claimed yet.

## Goal
Predict whether a telecom customer will churn (`Churn = Yes`) using the IBM Telco Customer Churn sample dataset. Demonstrate data validation, categorical preprocessing, leakage-safe pipelines, baseline classification, and reproducible evaluation.

## Dataset
Download `Telco-Customer-Churn.csv` from [IBM's archived official sample repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) and save it to `data/Telco-Customer-Churn.csv`.

The data are fictional sample data. Do not describe predictions as results from a real telecom operator.

## Run on macOS
```bash
cd customer-churn-ml-pipeline
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train.py
```

## What the script does
1. Loads the CSV, checks columns, and converts `TotalCharges` to a numeric feature.
2. Splits features/target into stratified train/test sets.
3. Builds a scikit-learn Pipeline with training-only imputers, scaling, and one-hot encoding.
4. Trains logistic regression and evaluates accuracy, precision, recall, F1, ROC-AUC and confusion matrix.
5. Saves metrics to `outputs/metrics.json` and the fitted pipeline to `outputs/churn_pipeline.joblib`.

## Next improvements
- Add EDA notebook and plotted churn distribution.
- Compare RandomForestClassifier against the baseline using cross-validation.
- Analyze false negatives and precision-recall tradeoffs.
- Write more unit tests and a short findings report.

## Ethical and reproducibility notes
Sample data are educational, not a real-world predictive deployment. Do not include customer personal identifiers as predictive features. Performance metrics must be generated from an actual run, not invented.
