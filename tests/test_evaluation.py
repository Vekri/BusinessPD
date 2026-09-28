"""Holdout metrics use probabilities, and Gini follows from ROC-AUC."""

import json

import numpy as np
import pandas as pd

from src.data_generator import generate_business_credit
from src.evaluate import calibration_table, evaluate_frame, evaluate_scores, ks_statistic
from src.features import add_features
from src.preprocessing import clean_business_credit
from src.split import split_frame
from src.train import fit_model


def test_gini_is_twice_auc_minus_one():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    report = evaluate_scores(y_true, scores, n_bins=2)
    assert report["gini"] == report["roc_auc"] * 2 - 1


def test_ks_on_separated_scores():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y_true, scores) == 1.0


def test_calibration_gap_is_predicted_minus_observed():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    table = calibration_table(y_true, scores, n_bins=2)
    assert list(table["bin"]) == [1, 2]
    assert table["gap"].iloc[0] == table["mean_predicted_pd"].iloc[0] - table["observed_default_rate"].iloc[0]


def test_holdout_report_ranks_defaults(tmp_path):
    clean = clean_business_credit(generate_business_credit(n_records=800, seed=15)).clean
    train, test = split_frame(add_features(clean), seed=15)
    report = evaluate_frame(test, fit_model(train))

    assert report["roc_auc"] > 0.7
    assert 0 < report["ks"] <= 1
    matrix = report["confusion_matrix"]
    assert sum(matrix.values()) == len(test)
    assert report["calibration"]

    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["roc_auc"] == report["roc_auc"]
    assert "gini" in loaded
    assert isinstance(loaded["calibration"], list)
    assert "business_id" not in pd.DataFrame(loaded["calibration"]).columns
