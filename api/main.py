"""FastAPI service for probability-of-default scoring.

Run from the project root:

    python -m uvicorn api.main:app --port 8001

Swagger is at http://127.0.0.1:8001/docs
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError

from api import repository
from src.config import (
    CREDIT_SCORE_MAX,
    CREDIT_SCORE_MIN,
    DELINQ_MAX,
    DTI_MAX,
    EVALUATION_PATH,
    MODEL_PATH,
    RISK_BANDS,
    VERY_HIGH_RISK,
    YEARS_MAX,
)
from src.assess import build_assessment
from src.predict import score_record
from src.train import load_model, train_file

DEMO_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


class PredictRequest(BaseModel):
    business_id: str = Field(min_length=1, max_length=50)
    business_name: str | None = Field(default=None, max_length=200)
    industry: str | None = Field(default=None, max_length=100)
    annual_revenue: float = Field(gt=0)
    profit: float
    total_debt: float = Field(gt=0)
    total_assets: float = Field(gt=0)
    credit_score: int = Field(ge=CREDIT_SCORE_MIN, le=CREDIT_SCORE_MAX)
    dti: float = Field(ge=0, le=DTI_MAX)
    delinq_12m: int = Field(ge=0, le=DELINQ_MAX)
    years_in_business: int = Field(ge=0, le=YEARS_MAX)
    cash_flow: float
    loan_amount: float = Field(gt=0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "business_id": "B1001",
                    "annual_revenue": 750000,
                    "profit": 85000,
                    "total_debt": 300000,
                    "total_assets": 850000,
                    "credit_score": 680,
                    "dti": 0.42,
                    "delinq_12m": 1,
                    "years_in_business": 6,
                    "cash_flow": 110000,
                    "loan_amount": 200000,
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    business_id: str
    pd: float
    risk_class: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    database: str


def create_app(artifact: dict | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifact = artifact if artifact is not None else load_model(MODEL_PATH)
        yield

    app = FastAPI(
        title="Business PD Classification",
        version="1.0",
        lifespan=lifespan,
    )

    def current_artifact():
        loaded = getattr(app.state, "artifact", None)
        if loaded is None:
            loaded = artifact if artifact is not None else load_model(MODEL_PATH)
            app.state.artifact = loaded
        return loaded

    @app.get("/")
    def demo() -> FileResponse:
        return FileResponse(DEMO_PAGE)

    @app.post("/assess")
    def assess(body: PredictRequest, loaded: dict = Depends(current_artifact)) -> dict:
        record = body.model_dump()
        try:
            result = build_assessment(record, loaded)
            if result["ok"]:
                try:
                    repository.save_application(record, result)
                    result["steps"].insert(
                        -1,
                        {
                            "id": "record",
                            "title": "Saved to the book",
                            "summary": (
                                f"{record['business_id']} is stored as a customer, "
                                "a scored loan application, and a model prediction."
                            ),
                            "items": [
                                {"label": "Customer", "value": record["business_id"]},
                                {"label": "Application", "value": "scored"},
                                {"label": "Prediction", "value": result["model_version"]},
                            ],
                        },
                    )
                except (SQLAlchemyError, OSError, ImportError):
                    result["steps"].insert(
                        -1,
                        {
                            "id": "record",
                            "title": "Score kept on this page",
                            "summary": (
                                f"{record['business_id']} is scored. This deployment has no "
                                "database connection, so the customer, application, and "
                                "prediction stay on this page."
                            ),
                            "items": [
                                {"label": "Customer", "value": "on this page"},
                                {"label": "Application", "value": "on this page"},
                                {"label": "Prediction", "value": result["model_version"]},
                            ],
                        },
                    )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        database = "ok" if repository.ping() else "unavailable"
        loaded = current_artifact()
        return HealthResponse(
            status="ok" if database == "ok" else "degraded",
            model_version=loaded["model_version"],
            database=database,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, loaded: dict = Depends(current_artifact)) -> PredictResponse:
        record = body.model_dump()
        try:
            scored = score_record(record, loaded)
            repository.save_application(record, scored)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PredictResponse(**scored)

    @app.get("/business/{business_id}")
    def business(business_id: str) -> dict:
        try:
            row = repository.get_business(business_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        if row is None:
            raise HTTPException(status_code=404, detail=f"Business not found: {business_id}")
        return row

    @app.get("/prediction/{business_id}", response_model=PredictResponse)
    def prediction(business_id: str) -> PredictResponse:
        try:
            row = repository.get_latest_prediction(business_id)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        if row is None:
            raise HTTPException(status_code=404, detail=f"Prediction not found: {business_id}")
        return PredictResponse(
            business_id=row["business_id"],
            pd=row["pd"],
            risk_class=row["risk_class"],
            model_version=row["model_version"],
        )

    @app.get("/businesses")
    def businesses(name: str = "", industry: str = "", limit: int = 20) -> dict:
        try:
            rows = repository.search_businesses(name=name, industry=industry, limit=limit)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc
        return {"count": len(rows), "businesses": rows}

    @app.get("/portfolio")
    def portfolio(min_pd: float = 0.15, limit: int = 20) -> dict:
        if not 0 <= min_pd <= 1:
            raise HTTPException(status_code=400, detail="min_pd must be between 0 and 1")
        try:
            return repository.portfolio(min_pd=min_pd, limit=limit)
        except SQLAlchemyError as exc:
            raise HTTPException(status_code=503, detail="Database is unavailable") from exc

    @app.get("/model")
    def model_information(loaded: dict = Depends(current_artifact)) -> dict:
        info = {
            "model_version": loaded["model_version"],
            "algorithm": "LogisticRegression",
            "class_weight": "balanced",
            "features": loaded.get("features", []),
            "train_rows": loaded.get("train_rows"),
            "risk_bands": [
                {"pd_below": RISK_BANDS[0][0], "risk_class": RISK_BANDS[0][1]},
                {"pd_below": RISK_BANDS[1][0], "risk_class": RISK_BANDS[1][1]},
                {"pd_at_or_below": RISK_BANDS[2][0], "risk_class": RISK_BANDS[2][1]},
                {"pd_above": RISK_BANDS[2][0], "risk_class": VERY_HIGH_RISK},
            ],
        }
        if EVALUATION_PATH.exists():
            report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
            info["holdout"] = {
                key: report[key]
                for key in ("rows", "roc_auc", "gini", "ks", "brier", "calibration_mae")
                if key in report
            }
        return info

    @app.post("/train")
    def train() -> dict:
        try:
            trained = train_file()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        app.state.artifact = trained
        return {
            "model_version": trained["model_version"],
            "train_rows": trained["train_rows"],
            "converged": trained["converged"],
        }

    return app


app = create_app()
