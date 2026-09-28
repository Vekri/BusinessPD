"""Risk bands follow the project cutoffs, and scoring returns a PD."""

from src.config import MODEL_FEATURES
from src.data_generator import generate_business_credit
from src.features import add_features
from src.predict import load_business, risk_class, score_record
from src.preprocessing import clean_business_credit
from src.train import fit_model


def test_risk_bands_at_the_stated_edges():
    assert risk_class(0.0499) == "Low Risk"
    assert risk_class(0.05) == "Moderate Risk"
    assert risk_class(0.0999) == "Moderate Risk"
    assert risk_class(0.10) == "High Risk"
    assert risk_class(0.20) == "High Risk"
    assert risk_class(0.2001) == "Very High Risk"


def test_score_record_from_raw_application():
    clean = clean_business_credit(generate_business_credit(n_records=400, seed=21)).clean
    artifact = fit_model(add_features(clean))
    record = {
        "business_id": "B1001",
        "annual_revenue": 750000,
        "profit": 85000,
        "total_debt": 300000,
        "total_assets": 850000,
        "credit_score": 680,
        "dti": 0.42,
        "delinq_12m": 1,
        "years_in_business": 6,
        "cash_flow": 110000,
        "loan_amount": 200000,
    }
    response = score_record(record, artifact)
    assert response["business_id"] == "B1001"
    assert response["model_version"] == artifact["model_version"]
    assert 0 < response["pd"] < 1
    assert response["risk_class"] == risk_class(response["pd"])


def test_stored_model_features_score_without_raw_amounts(tmp_path):
    clean = clean_business_credit(generate_business_credit(n_records=300, seed=22)).clean
    featured = add_features(clean)
    artifact = fit_model(featured)
    row = featured.iloc[0]
    stored = {"business_id": row["business_id"], **{column: row[column] for column in MODEL_FEATURES}}
    response = score_record(stored, artifact)
    assert response["business_id"] == row["business_id"]
    assert response["risk_class"] == risk_class(response["pd"])

    source = tmp_path / "features.csv"
    featured.to_csv(source, index=False)
    loaded = load_business(str(row["business_id"]), source)
    assert loaded["business_id"] == row["business_id"]
