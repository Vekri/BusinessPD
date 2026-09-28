"""The loader builds one customer, application, and prediction per business."""

from datetime import date

from src.config import VERY_HIGH_RISK
from src.data_generator import generate_business_credit
from src.features import add_features
from src.load import CUSTOMER_COLUMNS, build_load_frames
from src.predict import risk_class
from src.preprocessing import clean_business_credit
from src.train import fit_model

ALLOWED_RISK = {"Low Risk", "Moderate Risk", "High Risk", VERY_HIGH_RISK}


def test_load_frames_line_up():
    clean = clean_business_credit(generate_business_credit(n_records=200, seed=31)).clean
    featured = add_features(clean)
    artifact = fit_model(featured)
    customers, applications, predictions = build_load_frames(
        featured,
        artifact,
        application_date=date(2026, 9, 28),
    )

    assert len(customers) == len(applications) == len(predictions) == 200
    assert list(customers.columns) == CUSTOMER_COLUMNS
    assert customers["business_id"].tolist() == applications["business_id"].tolist()
    assert customers["business_id"].tolist() == predictions["business_id"].tolist()
    assert applications["status"].eq("submitted").all()
    assert applications["application_date"].eq(date(2026, 9, 28)).all()
    assert predictions["pd"].between(0, 1).all()
    assert set(predictions["risk_class"]).issubset(ALLOWED_RISK)
    assert predictions["risk_class"].tolist() == [risk_class(value) for value in predictions["pd"]]
    assert predictions["model_version"].eq(artifact["model_version"]).all()
