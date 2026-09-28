"""The logistic model fits on train rows only and reloads as a scoring pipeline."""

import numpy as np
import pandas as pd

from src.config import MODEL_FEATURES, MODEL_VERSION, TARGET_COLUMN
from src.data_generator import generate_business_credit
from src.features import add_features
from src.preprocessing import clean_business_credit
from src.split import split_frame
from src.train import coefficient_table, fit_model, load_model, save_model, training_matrix


def _train_frame(n_records: int = 400, seed: int = 11) -> pd.DataFrame:
    clean = clean_business_credit(generate_business_credit(n_records=n_records, seed=seed)).clean
    featured = add_features(clean)
    train, _test = split_frame(featured, seed=seed)
    return train


def test_training_matrix_uses_model_features_only():
    train = _train_frame()
    features, target = training_matrix(train)
    assert list(features.columns) == MODEL_FEATURES
    assert "business_id" not in features.columns
    assert set(target.unique()) <= {0, 1}


def test_fit_scores_are_probabilities_and_artifact_reloads(tmp_path):
    train = _train_frame()
    artifact = fit_model(train)
    path = save_model(artifact, tmp_path / "pd_model.joblib")
    loaded = load_model(path)

    assert loaded["model_version"] == MODEL_VERSION
    assert loaded["features"] == MODEL_FEATURES
    assert loaded["converged"] is True
    assert loaded["train_rows"] == len(train)

    features, target = training_matrix(train)
    probabilities = loaded["pipeline"].predict_proba(features)[:, 1]
    assert probabilities.shape == (len(train),)
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert list(loaded["pipeline"].named_steps["model"].classes_) == [0, 1]
    assert set(coefficient_table(loaded)["feature"]) == set(MODEL_FEATURES)
    assert target.isin([0, 1]).all()
