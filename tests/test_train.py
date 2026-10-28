import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train import prepare_data


def test_prepare_data_encodes_target_and_drops_id():
    df = pd.DataFrame({
        "customerID": ["a", "b"],
        "TotalCharges": ["10.5", " "],
        "Churn": ["Yes", "No"],
        "tenure": [1, 0],
    })
    X, y = prepare_data(df)
    assert y.tolist() == [1, 0]
    assert "customerID" not in X.columns
    assert pd.isna(X.loc[1, "TotalCharges"])


def test_prepare_data_rejects_bad_target():
    df = pd.DataFrame({
        "customerID": ["a"], "TotalCharges": ["10"], "Churn": ["Unknown"]
    })
    with pytest.raises(ValueError):
        prepare_data(df)
