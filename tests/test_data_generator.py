"""Checks that the synthetic book has the risk relationships we intended."""

import pandas as pd

from src.config import FEATURE_COLUMNS, ID_COLUMNS, N_RECORDS, TARGET_COLUMN
from src.data_generator import generate_business_credit, risk_correlations


def test_dataset_shape_and_domain():
    frame = generate_business_credit(n_records=N_RECORDS, seed=42)

    assert list(frame.columns) == ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(frame) == N_RECORDS
    assert frame["business_id"].is_unique
    assert frame.isna().sum().sum() == 0
    assert set(frame[TARGET_COLUMN].unique()) <= {0, 1}
    assert frame["credit_score"].between(300, 850).all()
    assert (frame["annual_revenue"] > 0).all()
    assert (frame["total_assets"] > 0).all()
    assert (frame["total_debt"] > 0).all()
    assert (frame["loan_amount"] > 0).all()
    assert frame["dti"].between(0, 2).all()
    assert (frame["delinq_12m"] >= 0).all()
    assert (frame["years_in_business"] >= 1).all()


def test_default_rate_is_learnable_not_degenerate():
    frame = generate_business_credit()
    rate = frame[TARGET_COLUMN].mean()
    assert 0.08 <= rate <= 0.40


def test_risk_drivers_point_the_right_way():
    frame = generate_business_credit()
    corr = risk_correlations(frame)

    assert corr["dti"] > 0.15
    assert corr["total_debt"] > 0.05
    assert corr["delinq_12m"] > 0.15
    assert corr["credit_score"] < -0.15
    assert corr["cash_flow"] < -0.05


def test_saved_csv_matches_generator(tmp_path):
    frame = generate_business_credit(n_records=200, seed=7)
    path = tmp_path / "business_credit.csv"
    frame.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert loaded.shape == frame.shape
    assert loaded["business_id"].tolist() == frame["business_id"].tolist()
