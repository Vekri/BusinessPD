"""Train the logistic probability-of-default model on the training split only.

Revenue, debt, and credit score sit on very different scales, so the saved
artifact is a pipeline: standardize using the training set, then logistic
regression. Coefficients are the change in log-odds for a one-standard-deviation
move in that feature. The test file is not read here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODEL_FEATURES,
    MODEL_PATH,
    MODEL_VERSION,
    TARGET_COLUMN,
    TRAIN_PATH,
)


def training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and the default target."""
    required = MODEL_FEATURES + [TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing training columns: {', '.join(missing)}")
    features = frame.loc[:, MODEL_FEATURES].copy()
    target = frame[TARGET_COLUMN].astype(int)
    if features.isna().any().any():
        raise ValueError("Training features contain missing values")
    if set(target.unique()) - {0, 1}:
        raise ValueError("Target must be 0 or 1")
    return features, target


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def fit_model(frame: pd.DataFrame) -> dict:
    """Fit on one frame and return the artifact saved to disk."""
    features, target = training_matrix(frame)
    pipeline = build_pipeline()
    pipeline.fit(features, target)
    estimator = pipeline.named_steps["model"]
    return {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "features": list(MODEL_FEATURES),
        "train_rows": int(len(frame)),
        "target": TARGET_COLUMN,
        "converged": bool(estimator.n_iter_[0] < estimator.max_iter),
        "iterations": int(estimator.n_iter_[0]),
    }


def coefficient_table(artifact: dict) -> pd.DataFrame:
    """Log-odds coefficients on the standardized features, largest first."""
    estimator = artifact["pipeline"].named_steps["model"]
    table = pd.DataFrame(
        {
            "feature": artifact["features"],
            "coefficient": estimator.coef_[0],
        }
    )
    table["abs_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient")
    table["coefficient"] = table["coefficient"].round(4)
    return table.reset_index(drop=True)


def save_model(artifact: dict, path: Path | str = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_model(path: Path | str = MODEL_PATH) -> dict:
    return joblib.load(path)


def train_file(source: Path | str = TRAIN_PATH, output: Path | str = MODEL_PATH) -> dict:
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Training file not found: {source}. Run python -m src.split first.")
    artifact = fit_model(pd.read_csv(source))
    save_model(artifact, output)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the logistic PD model")
    parser.add_argument("--input", type=str, default=str(TRAIN_PATH))
    parser.add_argument("--output", type=str, default=str(MODEL_PATH))
    args = parser.parse_args()

    artifact = train_file(args.input, args.output)
    print(f"Wrote {artifact['model_version']} to {args.output}")
    print(f"Train rows: {artifact['train_rows']:,}")
    print(f"Iterations: {artifact['iterations']} (converged: {artifact['converged']})")
    print(f"Intercept: {artifact['pipeline'].named_steps['model'].intercept_[0]:.4f}")
    print("Standardized coefficients:")
    print(coefficient_table(artifact).to_string(index=False))


if __name__ == "__main__":
    main()
