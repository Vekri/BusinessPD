"""Holdout evaluation of the saved probability-of-default model.

Classification metrics use a 0.5 probability cutoff. ROC-AUC, Gini, KS, and the
calibration table use the probability itself. The model is not refit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CALIBRATION_BINS,
    DECISION_THRESHOLD,
    EVALUATION_PATH,
    MODEL_PATH,
    TEST_PATH,
)
from src.train import load_model, training_matrix


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximum gap between the Good and Bad cumulative score distributions."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    y_sorted = y_true[order]
    n_bad = int(np.sum(y_sorted == 1))
    n_good = int(np.sum(y_sorted == 0))
    if n_bad == 0 or n_good == 0:
        raise ValueError("KS requires both default classes")
    cdf_bad = np.cumsum(y_sorted == 1) / n_bad
    cdf_good = np.cumsum(y_sorted == 0) / n_good
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def calibration_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    """Compare average predicted PD with the observed default rate by score bin."""
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "pd": np.asarray(scores, dtype=float)})
    frame["bin"] = pd.qcut(frame["pd"], q=n_bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("y", "size"),
            mean_predicted_pd=("pd", "mean"),
            observed_default_rate=("y", "mean"),
        )
        .reset_index(drop=True)
    )
    table.insert(0, "bin", np.arange(1, len(table) + 1))
    table["gap"] = table["mean_predicted_pd"] - table["observed_default_rate"]
    return table


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = DECISION_THRESHOLD,
    n_bins: int = CALIBRATION_BINS,
) -> dict:
    """Score one labeled set. `scores` are predicted probabilities of default."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    auc = float(roc_auc_score(y_true, scores))
    bins = calibration_table(y_true, scores, n_bins=n_bins)
    weights = bins["count"].to_numpy()
    calibration_mae = float(np.average(np.abs(bins["gap"]), weights=weights))
    return {
        "rows": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": auc,
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "calibration_mae": calibration_mae,
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
        "calibration": bins.round(4).to_dict(orient="records"),
    }


def evaluate_frame(frame: pd.DataFrame, artifact: dict) -> dict:
    features, target = training_matrix(frame)
    scores = artifact["pipeline"].predict_proba(features)[:, 1]
    report = evaluate_scores(target.to_numpy(), scores)
    report["model_version"] = artifact["model_version"]
    report["dataset"] = "test"
    return report


def _print_report(report: dict) -> None:
    print(f"Model: {report['model_version']}")
    print(f"Holdout rows: {report['rows']:,}")
    print(f"Decision threshold: {report['threshold']:.2f}")
    print(f"Accuracy:  {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall:    {report['recall']:.4f}")
    print(f"F1:        {report['f1']:.4f}")
    print(f"ROC-AUC:   {report['roc_auc']:.4f}")
    print(f"Gini:      {report['gini']:.4f}")
    print(f"KS:        {report['ks']:.4f}")
    print(f"Brier:     {report['brier']:.4f}")
    print(f"Calibration MAE: {report['calibration_mae']:.4f}")
    matrix = report["confusion_matrix"]
    print("Confusion matrix (rows are actual Good/Bad, columns are predicted Good/Bad):")
    print(f"  TN {matrix['true_negative']:5d}   FP {matrix['false_positive']:5d}")
    print(f"  FN {matrix['false_negative']:5d}   TP {matrix['true_positive']:5d}")
    print("Calibration (mean predicted PD vs observed default rate):")
    print(pd.DataFrame(report["calibration"]).to_string(index=False))


def evaluate_file(
    test_path: Path | str = TEST_PATH,
    model_path: Path | str = MODEL_PATH,
    output: Path | str = EVALUATION_PATH,
) -> dict:
    test_path = Path(test_path)
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run python -m src.train first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}. Run python -m src.split first.")
    report = evaluate_frame(pd.read_csv(test_path), load_model(model_path))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the PD model on the holdout")
    parser.add_argument("--test", type=str, default=str(TEST_PATH))
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--output", type=str, default=str(EVALUATION_PATH))
    args = parser.parse_args()

    report = evaluate_file(args.test, args.model, args.output)
    _print_report(report)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
