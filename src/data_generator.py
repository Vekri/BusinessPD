"""Synthetic business-credit data for probability-of-default modeling.

The generator builds 10,000 private businesses and assigns default labels from
a logistic function of observable risk drivers:

- higher DTI, debt, and 12-month delinquency raise default risk
- lower credit score raises default risk
- higher cash flow lowers default risk

Labels are stochastic, so the mapping is strong enough to learn and noisy
enough that it is not a deterministic rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    INDUSTRIES,
    N_RECORDS,
    RANDOM_SEED,
    TARGET_COLUMN,
)

_NAME_FIRST = [
    "North",
    "Summit",
    "Harbor",
    "Pioneer",
    "Cedar",
    "Bright",
    "Metro",
    "Valley",
    "Oak",
    "Silver",
    "Copper",
    "Lakeside",
]
_NAME_SECOND = [
    "Ridge",
    "Works",
    "Partners",
    "Supply",
    "Holdings",
    "Trade",
    "Field",
    "Forge",
    "Line",
    "Point",
]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def generate_business_credit(
    n_records: int = N_RECORDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a synthetic business-credit frame with a binary default target."""
    if n_records < 1:
        raise ValueError("n_records must be at least 1")

    rng = np.random.default_rng(seed)

    business_id = [f"B{i:05d}" for i in range(1, n_records + 1)]
    industry = rng.choice(INDUSTRIES, size=n_records)
    first = rng.choice(_NAME_FIRST, size=n_records)
    second = rng.choice(_NAME_SECOND, size=n_records)
    business_name = [
        f"{a} {b} {ind}" for a, b, ind in zip(first, second, industry)
    ]

    # Log-revenue centered near a mid-market firm (~$500k).
    annual_revenue = np.exp(rng.normal(13.15, 0.55, n_records))
    annual_revenue = np.clip(annual_revenue, 80_000.0, 8_000_000.0)

    years_in_business = rng.integers(1, 36, size=n_records)

    profit_margin = np.clip(rng.normal(0.11, 0.08, n_records), -0.20, 0.45)
    profit = annual_revenue * profit_margin

    asset_turnover = rng.uniform(0.35, 1.6, size=n_records)
    total_assets = np.clip(annual_revenue / asset_turnover, 50_000.0, None)

    # Debt/assets. Beta keeps most firms below 1.0 and still allows leverage.
    debt_to_asset = np.clip(rng.beta(2.2, 3.4, size=n_records) * 1.15, 0.02, 1.35)
    total_debt = total_assets * debt_to_asset

    # DTI tracks debt/revenue, with a small measurement gap like the sample rows.
    dti = np.clip(
        (total_debt / annual_revenue) * rng.normal(0.92, 0.06, n_records),
        0.02,
        1.80,
    )

    cash_flow_margin = np.clip(rng.normal(0.09, 0.07, n_records), -0.25, 0.40)
    cash_flow = annual_revenue * cash_flow_margin

    credit_score = (
        710
        - 160.0 * (dti - 0.40)
        - 40.0 * (debt_to_asset - 0.45)
        + 1.2 * (years_in_business - 8)
        + 40.0 * profit_margin
        + rng.normal(0.0, 28.0, n_records)
    )
    credit_score = np.clip(np.rint(credit_score), 300, 850).astype(int)

    # More delinquency when DTI is high and the bureau score is weak.
    delinq_lambda = np.exp(
        -1.55 + 2.15 * (dti - 0.35) - 0.006 * (credit_score - 680)
    )
    delinq_12m = rng.poisson(np.clip(delinq_lambda, 0.02, 5.0))

    loan_fraction = rng.uniform(0.08, 0.45, size=n_records)
    loan_amount = annual_revenue * loan_fraction

    # Coverage is the banking ratio. The cash-flow z-score is separate so the
    # raw cash_flow column itself moves default risk, not only the ratio.
    cash_flow_coverage = cash_flow / np.maximum(total_debt, 1_000.0)
    cash_flow_z = (cash_flow - np.mean(cash_flow)) / np.std(cash_flow)

    logit = (
        -1.70
        + 2.35 * (dti - 0.40)
        + 1.55 * (debt_to_asset - 0.45)
        + 0.48 * delinq_12m
        - 0.011 * (credit_score - 680)
        - 0.45 * np.clip(cash_flow_coverage, -0.4, 1.5)
        - 0.85 * cash_flow_z
        - 0.025 * (years_in_business - 8)
        + rng.normal(0.0, 0.40, n_records)
    )
    default_probability = _sigmoid(logit)
    default = rng.binomial(1, default_probability)

    frame = pd.DataFrame(
        {
            "business_id": business_id,
            "business_name": business_name,
            "industry": industry,
            "annual_revenue": np.round(annual_revenue, 2),
            "profit": np.round(profit, 2),
            "total_debt": np.round(total_debt, 2),
            "total_assets": np.round(total_assets, 2),
            "credit_score": credit_score,
            "dti": np.round(dti, 4),
            "delinq_12m": delinq_12m.astype(int),
            "years_in_business": years_in_business.astype(int),
            "cash_flow": np.round(cash_flow, 2),
            "loan_amount": np.round(loan_amount, 2),
            "default": default.astype(int),
        }
    )
    return frame.loc[:, ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]]


def save_dataset(frame: pd.DataFrame, path=DATA_PATH) -> None:
    """Write the dataset to CSV, creating the data directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def risk_correlations(frame: pd.DataFrame) -> dict[str, float]:
    """Pearson correlation of each intended risk driver with default."""
    drivers = ["dti", "total_debt", "delinq_12m", "credit_score", "cash_flow"]
    return {name: float(frame[name].corr(frame[TARGET_COLUMN])) for name in drivers}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic business credit data")
    parser.add_argument("--rows", type=int, default=N_RECORDS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=str, default=str(DATA_PATH))
    args = parser.parse_args()

    frame = generate_business_credit(n_records=args.rows, seed=args.seed)
    output_path = Path(args.output)
    save_dataset(frame, output_path)

    default_rate = float(frame[TARGET_COLUMN].mean())
    correlations = risk_correlations(frame)
    print(f"Wrote {len(frame):,} rows to {output_path}")
    print(f"Default rate: {default_rate:.1%}")
    print("Correlation with default:")
    for name, value in correlations.items():
        print(f"  {name:16} {value:+.3f}")


if __name__ == "__main__":
    main()
