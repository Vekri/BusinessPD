"""Run the SQL reports in sql/queries.sql against the local book."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from src.config import PROJECT_ROOT
from src.db import get_engine

QUERIES_PATH = PROJECT_ROOT / "sql" / "queries.sql"


def load_queries(path: Path | str = QUERIES_PATH) -> dict[str, str]:
    """Return named SQL statements from a queries file."""
    queries: dict[str, str] = {}
    name: str | None = None
    lines: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("-- name:"):
            if name and lines:
                queries[name] = "\n".join(lines).strip().rstrip(";")
            name = line.split(":", 1)[1].strip()
            lines = []
            continue
        if name is not None and not line.startswith("--"):
            lines.append(line)
    if name and lines:
        queries[name] = "\n".join(lines).strip().rstrip(";")
    if not queries:
        raise ValueError(f"No named queries found in {path}")
    return queries


def run_query(sql: str, limit: int | None = None) -> pd.DataFrame:
    statement = f"SELECT * FROM ({sql}) AS report"
    if limit is not None:
        statement = f"{statement} LIMIT {int(limit)}"
    with get_engine().connect() as connection:
        return pd.read_sql(text(statement), connection)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print PD portfolio reports")
    parser.add_argument("--sample", type=int, default=5, help="Rows to show from the detail queries")
    args = parser.parse_args()

    queries = load_queries()
    summary = run_query(queries["risk_summary"])
    above = run_query(queries["pd_above_15"])
    sample = run_query(queries["latest_scores"], limit=args.sample)

    print("Risk summary")
    print(summary.to_string(index=False))
    print()
    print(f"Businesses with PD above 15%: {len(above):,}")
    print(above.head(args.sample).to_string(index=False))
    print()
    print(f"Latest scores (first {args.sample})")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
