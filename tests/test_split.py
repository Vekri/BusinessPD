"""The holdout is stratified, disjoint, and reproducible."""

import pandas as pd

from src.config import N_RECORDS, TARGET_COLUMN, TEST_SIZE
from src.data_generator import generate_business_credit
from src.features import add_features
from src.preprocessing import clean_business_credit
from src.split import split_frame, write_split


def _featured(n_records: int = N_RECORDS, seed: int = 42) -> pd.DataFrame:
    clean = clean_business_credit(generate_business_credit(n_records=n_records, seed=seed)).clean
    return add_features(clean)


def test_split_sizes_and_no_shared_ids():
    frame = _featured()
    train, test = split_frame(frame)

    assert len(train) == int(N_RECORDS * (1 - TEST_SIZE))
    assert len(test) == int(N_RECORDS * TEST_SIZE)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"]))
    assert set(train["business_id"]) | set(test["business_id"]) == set(frame["business_id"])


def test_default_rate_is_preserved():
    frame = _featured()
    train, test = split_frame(frame)
    full_rate = frame[TARGET_COLUMN].mean()
    assert abs(train[TARGET_COLUMN].mean() - full_rate) < 0.01
    assert abs(test[TARGET_COLUMN].mean() - full_rate) < 0.01


def test_same_seed_returns_the_same_businesses():
    frame = _featured(n_records=500, seed=9)
    first_train, first_test = split_frame(frame, seed=9)
    second_train, second_test = split_frame(frame, seed=9)
    assert first_train["business_id"].tolist() == second_train["business_id"].tolist()
    assert first_test["business_id"].tolist() == second_test["business_id"].tolist()


def test_write_split_round_trip(tmp_path):
    source = tmp_path / "features.csv"
    _featured(n_records=100, seed=6).to_csv(source, index=False)
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train, test = write_split(source, train_path, test_path)
    assert len(pd.read_csv(train_path)) == len(train) == 80
    assert len(pd.read_csv(test_path)) == len(test) == 20
