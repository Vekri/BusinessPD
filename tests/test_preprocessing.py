"""Cleaning keeps valid businesses and drops rows that break the data contract."""

import pandas as pd
import pytest

from src.config import DATA_PATH, N_RECORDS, REQUIRED_COLUMNS
from src.data_generator import generate_business_credit
from src.preprocessing import clean_business_credit, clean_file


def _bad_row(frame: pd.DataFrame, **changes) -> pd.DataFrame:
    row = frame.iloc[[0]].copy()
    for column, value in changes.items():
        row[column] = value
    return pd.concat([frame, row], ignore_index=True)


def test_generated_book_passes_clean():
    result = clean_business_credit(generate_business_credit())
    assert result.report.rows_in == N_RECORDS
    assert result.report.rows_kept == N_RECORDS
    assert result.report.rows_rejected == 0
    assert result.report.reasons == {}
    assert list(result.clean.columns) == REQUIRED_COLUMNS


def test_saved_csv_passes_clean():
    result = clean_business_credit(pd.read_csv(DATA_PATH))
    assert result.report.rows_kept == N_RECORDS
    assert result.report.rows_rejected == 0


def test_missing_column_stops_the_run():
    frame = generate_business_credit(n_records=5, seed=1).drop(columns=["dti"])
    with pytest.raises(ValueError, match="dti"):
        clean_business_credit(frame)


def test_invalid_rows_are_rejected_with_reasons():
    frame = generate_business_credit(n_records=5, seed=2)
    frame = _bad_row(frame, business_id="B00999", credit_score=900, annual_revenue=-10)
    frame = _bad_row(frame, business_id="  B00998  ", business_name="  Kept Name  ")
    frame = _bad_row(frame, business_id="B00001", default=0)
    frame = _bad_row(frame, business_id="B00997", default=2)

    result = clean_business_credit(frame)

    assert result.report.rows_rejected == 3
    assert result.report.reasons["invalid_credit_score"] == 1
    assert result.report.reasons["invalid_annual_revenue"] == 1
    assert result.report.reasons["duplicate_business_id"] == 1
    assert result.report.reasons["invalid_default"] == 1
    assert result.clean["business_id"].is_unique
    assert "B00999" not in set(result.clean["business_id"])
    kept = result.clean.loc[result.clean["business_id"] == "B00998"].iloc[0]
    assert kept["business_name"] == "Kept Name"


def test_unknown_industry_is_kept_and_warned():
    frame = generate_business_credit(n_records=4, seed=3)
    frame.loc[0, "industry"] = "Marine"
    result = clean_business_credit(frame)
    assert result.report.rows_rejected == 0
    assert result.report.warnings["unknown_industry"] == 1


def test_clean_file_writes_outputs(tmp_path):
    source = tmp_path / "raw.csv"
    generate_business_credit(n_records=8, seed=4).to_csv(source, index=False)
    clean_path = tmp_path / "clean.csv"
    reject_path = tmp_path / "rejected.csv"
    result = clean_file(source, clean_path, reject_path)
    loaded = pd.read_csv(clean_path)
    assert len(loaded) == result.report.rows_kept == 8
    assert not reject_path.exists()
