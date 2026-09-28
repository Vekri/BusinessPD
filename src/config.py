"""Project paths and dataset contract for the business PD pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"
DEFAULT_DATABASE_URL = "postgresql+psycopg2://pd_user:pd_password@127.0.0.1:5434/business_pd"
DATA_PATH = DATA_DIR / "business_credit.csv"

N_RECORDS = 10_000
RANDOM_SEED = 42

ID_COLUMNS = [
    "business_id",
    "business_name",
    "industry",
]

FEATURE_COLUMNS = [
    "annual_revenue",
    "profit",
    "total_debt",
    "total_assets",
    "credit_score",
    "dti",
    "delinq_12m",
    "years_in_business",
    "cash_flow",
    "loan_amount",
]

TARGET_COLUMN = "default"
REQUIRED_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]

ENGINEERED_COLUMNS = [
    "debt_to_asset",
    "loan_to_revenue",
    "profit_margin",
    "cash_flow_coverage",
]
MODEL_FEATURES = FEATURE_COLUMNS + ENGINEERED_COLUMNS

# Educational labels only. A production model would use the institution's
# default definition and regulatory methodology.
GOOD_LABEL = 0
BAD_LABEL = 1

INDUSTRIES = [
    "Retail",
    "Manufacturing",
    "Construction",
    "Wholesale",
    "Professional Services",
    "Transportation",
    "Hospitality",
    "Healthcare",
    "Agriculture",
    "Technology",
]

# Row-level bounds. Values outside these ranges are rejected, not clipped.
CREDIT_SCORE_MIN = 300
CREDIT_SCORE_MAX = 850
DTI_MIN = 0.0
DTI_MAX = 5.0
DELINQ_MIN = 0
DELINQ_MAX = 24
YEARS_MIN = 0
YEARS_MAX = 100

CLEAN_PATH = DATA_DIR / "business_credit_clean.csv"
REJECT_PATH = DATA_DIR / "rejected_rows.csv"
FEATURES_PATH = DATA_DIR / "business_credit_features.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
TEST_SIZE = 0.20

MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "pd_model.joblib"
EVALUATION_PATH = MODEL_DIR / "evaluation.json"
MODEL_VERSION = "pd_model_v1"
DECISION_THRESHOLD = 0.50
CALIBRATION_BINS = 10

# Project bands, not a regulatory standard. Edges: 5% and 10% start the next
# band; 20% itself stays High Risk, and only a PD above 20% is Very High Risk.
RISK_BANDS = (
    (0.05, "Low Risk"),
    (0.10, "Moderate Risk"),
    (0.20, "High Risk"),
)
VERY_HIGH_RISK = "Very High Risk"
