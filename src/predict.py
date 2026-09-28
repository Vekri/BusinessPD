"""Score one business with the saved probability-of-default model.

A new application supplies the raw financial fields. Ratios are calculated with
the same formulas used in training. A row that already has the model features
is scored as stored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURES_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    RISK_BANDS,
    VERY_HIGH_RISK,
)
from src.features import add_features
from src.train import load_model


def risk_class(probability: float) -> str:
    """Map a probability of default to the project risk band."""
    low_max, low_label = RISK_BANDS[0]
    moderate_max, moderate_label = RISK_BANDS[1]
    high_max, high_label = RISK_BANDS[2]
    if probability < low_max:
        return low_label
    if probability < moderate_max:
        return moderate_label
    if probability <= high_max:
        return high_label
    return VERY_HIGH_RISK


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Use stored model features, or build the ratios from raw fields."""
    if all(column in frame.columns for column in MODEL_FEATURES):
        return frame.loc[:, MODEL_FEATURES]
    return add_features(frame).loc[:, MODEL_FEATURES]


def score_frame(frame: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Add pd and risk_class columns for each row."""
    features = model_inputs(frame)
    if features.isna().any().any():
        raise ValueError("Scoring features contain missing values")
    probabilities = artifact["pipeline"].predict_proba(features)[:, 1]
    scored = frame.copy()
    scored["pd"] = [round(float(value), 4) for value in probabilities]
    scored["risk_class"] = [risk_class(value) for value in scored["pd"]]
    scored["model_version"] = artifact["model_version"]
    return scored


def score_record(record: dict, artifact: dict) -> dict:
    """Return the PD response for one business."""
    scored = score_frame(pd.DataFrame([record]), artifact).iloc[0]
    response = {
        "pd": float(scored["pd"]),
        "risk_class": str(scored["risk_class"]),
        "model_version": str(scored["model_version"]),
    }
    if "business_id" in record and pd.notna(record["business_id"]):
        response = {"business_id": str(record["business_id"]), **response}
    return response


def load_business(business_id: str, source: Path | str = FEATURES_PATH) -> dict:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Feature file not found: {source}. Run python -m src.features first.")
    frame = pd.read_csv(source, dtype={"business_id": str})
    matches = frame.loc[frame["business_id"] == business_id]
    if matches.empty:
        raise ValueError(f"Business not found: {business_id}")
    return matches.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score probability of default for one business")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--business-id", type=str, help="Id in the feature file, for example B00002")
    group.add_argument("--input", type=str, help="JSON file with one application")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--data", type=str, default=str(FEATURES_PATH))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run python -m src.train first.")
    artifact = load_model(model_path)
    if args.business_id:
        record = load_business(args.business_id, args.data)
    else:
        record = json.loads(Path(args.input).read_text(encoding="utf-8"))
    print(json.dumps(score_record(record, artifact), indent=2))


if __name__ == "__main__":
    main()
