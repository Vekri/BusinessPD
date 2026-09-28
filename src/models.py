"""SQLAlchemy tables matching sql/schema.sql.

The SQL file is what PostgreSQL executes. These classes are the same column
contract for later API code.
"""

from sqlalchemy import Date, DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class BusinessCustomer(Base):
    __tablename__ = "business_customer"

    business_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    business_name: Mapped[str] = mapped_column(String(200))
    industry: Mapped[str] = mapped_column(String(100))
    annual_revenue: Mapped[float] = mapped_column(Numeric(18, 2))
    profit: Mapped[float] = mapped_column(Numeric(18, 2))
    total_debt: Mapped[float] = mapped_column(Numeric(18, 2))
    total_assets: Mapped[float] = mapped_column(Numeric(18, 2))
    credit_score: Mapped[int] = mapped_column(Integer)
    dti: Mapped[float] = mapped_column(Numeric(10, 4))
    delinq_12m: Mapped[int] = mapped_column(Integer)
    years_in_business: Mapped[int] = mapped_column(Integer)
    cash_flow: Mapped[float] = mapped_column(Numeric(18, 2))
    created_at: Mapped[str] = mapped_column(DateTime)


class LoanApplication(Base):
    __tablename__ = "loan_application"

    application_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("business_customer.business_id"))
    loan_amount: Mapped[float] = mapped_column(Numeric(18, 2))
    application_date: Mapped[str] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(30))


class ModelPrediction(Base):
    __tablename__ = "model_prediction"

    prediction_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[str] = mapped_column(ForeignKey("business_customer.business_id"))
    model_version: Mapped[str] = mapped_column(String(50))
    pd: Mapped[float] = mapped_column(Numeric(12, 8))
    risk_class: Mapped[str] = mapped_column(String(50))
    prediction_date: Mapped[str] = mapped_column(DateTime)
