"""REST API for the NileMart sales forecast.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nilemart import __version__
from nilemart.predict import Plan, forecast, load_model

app = FastAPI(title="NileMart Sales Forecast", version=__version__)


class PlanIn(BaseModel):
    ad_spend_k: float = Field(..., ge=0, description="Ad spend for the week, EGP thousand", examples=[80])
    discount_pct: float = Field(..., ge=0, le=100, description="Average discount, %", examples=[10])
    is_holiday_period: int = Field(..., ge=0, le=1, description="1 for Ramadan, Eid or year-end", examples=[0])
    competitor_promo: int = Field(..., ge=0, le=1, description="1 if the competitor runs a promotion", examples=[1])


@app.get("/health")
def health():
    _, meta = load_model()
    return {"status": "ok", "model": meta["model"], "version": meta["version"], "trained_at": meta["trained_at"]}


@app.post("/forecast")
def post_forecast(plan: PlanIn):
    return forecast(Plan(**plan.model_dump()))


@app.post("/forecast/batch")
def post_batch(plans: list[PlanIn]):
    return [forecast(Plan(**p.model_dump())) for p in plans]
