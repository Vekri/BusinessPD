"""Load customers, loan applications, and PD scores into PostgreSQL.

The feature file is the source. Each business becomes one customer, one
submitted application, and one prediction from the saved model. Re-running
this command rebuilds the tables from sql/schema.sql, then loads them again.
"""

from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
from sqlalchemy import text

from src.config import FEATURES_PATH, MODEL_PATH
from src.db import apply_schema, get_engine
from src.predict import score_frame
from src.train import load_model

CUSTOMER_COLUMNS = [
    "business_id",
    "business_name",
    "industry",
    "annual_revenue",
    "profit",
    "total_debt",
    "total_assets",
    "credit_score",
    "dti",
    "delinq_12m",
    "years_in_business",
    "cash_flow",
]


def build_load_frames(
    frame: pd.DataFrame,
    artifact: dict,
    application_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a scored book into the three tables."""
    scored = score_frame(frame, artifact)
    customers = scored.loc[:, CUSTOMER_COLUMNS].copy()
    applications = scored.loc[:, ["business_id", "loan_amount"]].copy()
    applications["application_date"] = application_date or date.today()
    applications["status"] = "submitted"
    predictions = scored.loc[:, ["business_id", "model_version", "pd", "risk_class"]].copy()
    return customers, applications, predictions


def load_frames(customers: pd.DataFrame, applications: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, int]:
    """Rebuild the schema and insert the three frames."""
    engine = get_engine()
    apply_schema(engine)
    customers.to_sql("business_customer", engine, if_exists="append", index=False, method="multi", chunksize=1000)
    applications.to_sql("loan_application", engine, if_exists="append", index=False, method="multi", chunksize=1000)
    predictions.to_sql("model_prediction", engine, if_exists="append", index=False, method="multi", chunksize=1000)
    with engine.connect() as connection:
        counts = {
            "business_customer": connection.execute(text("SELECT COUNT(*) FROM business_customer")).scalar_one(),
            "loan_application": connection.execute(text("SELECT COUNT(*) FROM loan_application")).scalar_one(),
            "model_prediction": connection.execute(text("SELECT COUNT(*) FROM model_prediction")).scalar_one(),
        }
    return {name: int(count) for name, count in counts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Load the scored book into PostgreSQL")
    parser.add_argument("--input", type=str, default=str(FEATURES_PATH))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    artifact = load_model(args.model)
    customers, applications, predictions = build_load_frames(frame, artifact)
    counts = load_frames(customers, applications, predictions)
    print(f"business_customer  {counts['business_customer']:,}")
    print(f"loan_application   {counts['loan_application']:,}")
    print(f"model_prediction   {counts['model_prediction']:,}")


if __name__ == "__main__":
    main()
