"""Database connection for the local PostgreSQL book."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from src.config import DEFAULT_DATABASE_URL, PROJECT_ROOT, SCHEMA_PATH

load_dotenv(PROJECT_ROOT / ".env")


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def get_engine() -> Engine:
    return create_engine(database_url(), connect_args={"connect_timeout": 3})


def apply_schema(engine: Engine | None = None) -> None:
    """Drop and recreate the three tables from sql/schema.sql."""
    engine = engine or get_engine()
    url = engine.url
    import psycopg2

    connection = psycopg2.connect(
        host=url.host or "localhost",
        port=url.port or 5432,
        dbname=url.database,
        user=url.username,
        password=url.password,
    )
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    finally:
        connection.close()
