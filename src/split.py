"""Stratified train/test split for the probability-of-default model.

The holdout keeps the same default rate as the full book, so evaluation is not
distorted by a test set that is mostly Good or mostly Bad. No model is fit here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES_PATH,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)


def split_frame(
    frame: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test frames with the same default mix as the input."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")
    if "business_id" not in frame.columns:
        raise ValueError("Missing business_id column")
    if frame["business_id"].duplicated().any():
        raise ValueError("business_id values must be unique before splitting")
    if frame[TARGET_COLUMN].nunique() < 2:
        raise ValueError("Both default classes are required for a stratified split")

    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[TARGET_COLUMN],
    )
    train = train.sort_values("business_id").reset_index(drop=True)
    test = test.sort_values("business_id").reset_index(drop=True)
    return train, test


def write_split(
    source: Path | str = FEATURES_PATH,
    train_path: Path | str = TRAIN_PATH,
    test_path: Path | str = TEST_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the feature file and write train.csv and test.csv."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"Feature file not found: {source}. Run python -m src.features first."
        )
    train, test = split_frame(pd.read_csv(source), test_size=test_size, seed=seed)
    train_path = Path(train_path)
    test_path = Path(test_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    return train, test


def _rate(frame: pd.DataFrame) -> float:
    return float(frame[TARGET_COLUMN].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the feature file into train and test")
    parser.add_argument("--input", type=str, default=str(FEATURES_PATH))
    parser.add_argument("--train-output", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--test-output", type=str, default=str(TEST_PATH))
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    train, test = write_split(
        args.input,
        args.train_output,
        args.test_output,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"Wrote {len(train):,} train rows to {args.train_output}")
    print(f"Wrote {len(test):,} test rows to {args.test_output}")
    print(f"Train default rate: {_rate(train):.1%}")
    print(f"Test default rate:  {_rate(test):.1%}")


if __name__ == "__main__":
    main()
