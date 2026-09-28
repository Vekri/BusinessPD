"""Banking ratios calculated after cleaning and before model training.

debt_to_asset       = total_debt / total_assets
loan_to_revenue     = loan_amount / annual_revenue
profit_margin       = profit / annual_revenue
cash_flow_coverage  = cash_flow / total_debt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import (
    CLEAN_PATH,
    ENGINEERED_COLUMNS,
    FEATURES_PATH,
    ID_COLUMNS,
    MODEL_FEATURES,
    TARGET_COLUMN,
)

_RATIO_INPUTS = [
    "annual_revenue",
    "profit",
    "total_debt",
    "total_assets",
    "cash_flow",
    "loan_amount",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the frame with the four engineered ratios appended."""
    missing = [column for column in _RATIO_INPUTS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns required for ratios: {', '.join(missing)}")

    work = frame.copy()
    if (work["annual_revenue"] <= 0).any() or (work["total_assets"] <= 0).any():
        raise ValueError(
            "annual_revenue and total_assets must be positive. Run preprocessing first."
        )
    if (work["total_debt"] <= 0).any():
        raise ValueError(
            "total_debt must be positive to compute cash_flow_coverage. Run preprocessing first."
        )

    work["debt_to_asset"] = (work["total_debt"] / work["total_assets"]).round(6)
    work["loan_to_revenue"] = (work["loan_amount"] / work["annual_revenue"]).round(6)
    work["profit_margin"] = (work["profit"] / work["annual_revenue"]).round(6)
    work["cash_flow_coverage"] = (work["cash_flow"] / work["total_debt"]).round(6)

    columns = [column for column in ID_COLUMNS + MODEL_FEATURES if column in work.columns]
    if TARGET_COLUMN in work.columns:
        columns.append(TARGET_COLUMN)
    return work.loc[:, columns].reset_index(drop=True)


def build_feature_file(
    source: Path | str = CLEAN_PATH,
    output: Path | str = FEATURES_PATH,
) -> pd.DataFrame:
    """Read the cleaned CSV, add ratios, and write the model-ready file."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"Clean file not found: {source}. Run python -m src.preprocessing first."
        )
    featured = add_features(pd.read_csv(source))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    featured.to_csv(output, index=False)
    return featured


def main() -> None:
    parser = argparse.ArgumentParser(description="Add banking ratios to the cleaned book")
    parser.add_argument("--input", type=str, default=str(CLEAN_PATH))
    parser.add_argument("--output", type=str, default=str(FEATURES_PATH))
    args = parser.parse_args()

    featured = build_feature_file(args.input, args.output)
    print(f"Wrote {len(featured):,} rows to {args.output}")
    print("Engineered ratios:")
    summary = featured[ENGINEERED_COLUMNS].agg(["min", "median", "max"]).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
