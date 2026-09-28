"""Engineered ratios match the banking definitions."""

import pandas as pd
import pytest

from src.config import ENGINEERED_COLUMNS, MODEL_FEATURES, N_RECORDS, TARGET_COLUMN
from src.data_generator import generate_business_credit
from src.features import add_features, build_feature_file
from src.preprocessing import clean_business_credit


def _sample(**overrides) -> pd.DataFrame:
    row = {
        "business_id": "B00001",
        "business_name": "Sample Works",
        "industry": "Retail",
        "annual_revenue": 100.0,
        "profit": 25.0,
        "total_debt": 40.0,
        "total_assets": 80.0,
        "credit_score": 700,
        "dti": 0.4,
        "delinq_12m": 0,
        "years_in_business": 5,
        "cash_flow": 10.0,
        "loan_amount": 20.0,
        "default": 0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_ratios_match_definitions():
    featured = add_features(_sample())
    row = featured.iloc[0]
    assert row["debt_to_asset"] == pytest.approx(0.5)
    assert row["loan_to_revenue"] == pytest.approx(0.2)
    assert row["profit_margin"] == pytest.approx(0.25)
    assert row["cash_flow_coverage"] == pytest.approx(0.25)
    assert list(featured.columns[-5:]) == ENGINEERED_COLUMNS + [TARGET_COLUMN]


def test_negative_profit_and_cash_flow_stay_negative():
    featured = add_features(_sample(profit=-10, cash_flow=-8))
    row = featured.iloc[0]
    assert row["profit_margin"] == pytest.approx(-0.1)
    assert row["cash_flow_coverage"] == pytest.approx(-0.2)


def test_zero_debt_is_rejected():
    with pytest.raises(ValueError, match="total_debt"):
        add_features(_sample(total_debt=0))


def test_cleaned_book_gets_finite_model_features():
    clean = clean_business_credit(generate_business_credit()).clean
    featured = add_features(clean)
    assert len(featured) == N_RECORDS
    assert featured[MODEL_FEATURES].isna().sum().sum() == 0
    assert featured[ENGINEERED_COLUMNS].replace([float("inf"), float("-inf")], pd.NA).isna().sum().sum() == 0


def test_feature_file_round_trip(tmp_path):
    source = tmp_path / "clean.csv"
    clean_business_credit(generate_business_credit(n_records=12, seed=5)).clean.to_csv(source, index=False)
    output = tmp_path / "features.csv"
    featured = build_feature_file(source, output)
    loaded = pd.read_csv(output)
    assert len(loaded) == len(featured) == 12
    assert set(ENGINEERED_COLUMNS).issubset(loaded.columns)
