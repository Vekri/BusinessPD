"""Clean and validate the business-credit training file.

Schema problems stop the run. Bad rows are removed and counted. Values are
not clipped: a credit score of 900 is a rejected row, not a score of 850.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_PATH,
    CREDIT_SCORE_MAX,
    CREDIT_SCORE_MIN,
    DATA_PATH,
    DELINQ_MAX,
    DELINQ_MIN,
    DTI_MAX,
    DTI_MIN,
    FEATURE_COLUMNS,
    ID_COLUMNS,
    INDUSTRIES,
    REJECT_PATH,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    YEARS_MAX,
    YEARS_MIN,
)

_MONEY_COLUMNS = [
    "annual_revenue",
    "profit",
    "total_debt",
    "total_assets",
    "cash_flow",
    "loan_amount",
]
_COUNT_COLUMNS = ["credit_score", "delinq_12m", "years_in_business", TARGET_COLUMN]
_POSITIVE_COLUMNS = ["annual_revenue", "total_assets", "loan_amount"]
_NON_NEGATIVE_COLUMNS = ["total_debt"]


@dataclass
class ValidationReport:
    """Counts from one cleaning pass."""

    rows_in: int
    rows_kept: int
    rows_rejected: int
    reasons: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        lines = [
            f"Rows in:       {self.rows_in:,}",
            f"Rows kept:     {self.rows_kept:,}",
            f"Rows rejected: {self.rows_rejected:,}",
        ]
        if self.reasons:
            lines.append("Reject reasons:")
            for name, count in self.reasons.items():
                lines.append(f"  {name:28} {count:,}")
        if self.warnings:
            lines.append("Warnings:")
            for name, count in self.warnings.items():
                lines.append(f"  {name:28} {count:,}")
        return lines


@dataclass
class CleanResult:
    clean: pd.DataFrame
    rejected: pd.DataFrame
    report: ValidationReport


def _whole_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric) & np.isclose(numeric, np.rint(numeric))


def _finite_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & np.isfinite(numeric)


def clean_business_credit(frame: pd.DataFrame) -> CleanResult:
    """Validate schema, drop bad rows, and return the kept book plus a report."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    work = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ID_COLUMNS:
        work[column] = work[column].astype("string").str.strip()

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    flags: dict[str, pd.Series] = {
        "missing_business_id": work["business_id"].isna() | (work["business_id"] == ""),
        "missing_business_name": work["business_name"].isna() | (work["business_name"] == ""),
        "missing_industry": work["industry"].isna() | (work["industry"] == ""),
        "invalid_profit": ~_finite_number(work["profit"]),
        "invalid_cash_flow": ~_finite_number(work["cash_flow"]),
        "invalid_dti": ~_finite_number(work["dti"])
        | (work["dti"] < DTI_MIN)
        | (work["dti"] > DTI_MAX),
        "invalid_credit_score": ~_whole_number(work["credit_score"])
        | (work["credit_score"] < CREDIT_SCORE_MIN)
        | (work["credit_score"] > CREDIT_SCORE_MAX),
        "invalid_delinq_12m": ~_whole_number(work["delinq_12m"])
        | (work["delinq_12m"] < DELINQ_MIN)
        | (work["delinq_12m"] > DELINQ_MAX),
        "invalid_years_in_business": ~_whole_number(work["years_in_business"])
        | (work["years_in_business"] < YEARS_MIN)
        | (work["years_in_business"] > YEARS_MAX),
        "invalid_default": ~_whole_number(work[TARGET_COLUMN])
        | ~work[TARGET_COLUMN].isin([0, 1]),
    }
    for column in _POSITIVE_COLUMNS:
        flags[f"invalid_{column}"] = ~_finite_number(work[column]) | (work[column] <= 0)
    for column in _NON_NEGATIVE_COLUMNS:
        flags[f"invalid_{column}"] = ~_finite_number(work[column]) | (work[column] < 0)

    id_present = ~flags["missing_business_id"]
    flags["duplicate_business_id"] = work["business_id"].duplicated(keep="first") & id_present

    reason_frame = pd.DataFrame(flags)
    rejected_mask = reason_frame.any(axis=1)
    reason_text = reason_frame.apply(
        lambda row: ";".join(name for name, flagged in row.items() if flagged),
        axis=1,
    )

    clean = work.loc[~rejected_mask].copy()
    for column in _COUNT_COLUMNS:
        clean[column] = clean[column].round().astype(int)
    for column in ("annual_revenue", "profit", "total_debt", "total_assets", "cash_flow", "loan_amount"):
        clean[column] = clean[column].round(2)
    clean["dti"] = clean["dti"].round(4)
    clean = clean.reset_index(drop=True)

    rejected = work.loc[rejected_mask].copy()
    rejected.insert(0, "source_row", rejected.index.astype(int) + 2)
    rejected["reject_reason"] = reason_text.loc[rejected_mask].to_numpy()
    rejected = rejected.reset_index(drop=True)

    known_industry = set(INDUSTRIES)
    unknown_industry = int((~clean["industry"].isin(known_industry)).sum()) if len(clean) else 0
    warnings = {"unknown_industry": unknown_industry} if unknown_industry else {}

    reasons = {
        name: int(reason_frame[name].sum())
        for name in reason_frame.columns
        if int(reason_frame[name].sum()) > 0
    }
    report = ValidationReport(
        rows_in=int(len(work)),
        rows_kept=int(len(clean)),
        rows_rejected=int(len(rejected)),
        reasons=reasons,
        warnings=warnings,
    )
    return CleanResult(clean=clean, rejected=rejected, report=report)


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_file(
    source: Path | str = DATA_PATH,
    clean_path: Path | str = CLEAN_PATH,
    reject_path: Path | str = REJECT_PATH,
) -> CleanResult:
    """Read a CSV, clean it, and write the kept rows."""
    result = clean_business_credit(load_raw(source))
    clean_path = Path(clean_path)
    reject_path = Path(reject_path)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    result.clean.to_csv(clean_path, index=False)
    if result.rejected.empty:
        if reject_path.exists():
            reject_path.unlink()
    else:
        result.rejected.to_csv(reject_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and validate business credit data")
    parser.add_argument("--input", type=str, default=str(DATA_PATH))
    parser.add_argument("--clean-output", type=str, default=str(CLEAN_PATH))
    parser.add_argument("--reject-output", type=str, default=str(REJECT_PATH))
    args = parser.parse_args()

    result = clean_file(args.input, args.clean_output, args.reject_output)
    print(f"Wrote {result.report.rows_kept:,} rows to {args.clean_output}")
    if result.report.rows_rejected:
        print(f"Wrote {result.report.rows_rejected:,} rejected rows to {args.reject_output}")
    for line in result.report.lines():
        print(line)


if __name__ == "__main__":
    main()
