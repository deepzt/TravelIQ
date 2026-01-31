from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.forecast import forecast_signal
from src.cancellation_risk import compute_cancellation_risk_table, lookup_cancellation_risk
from src.overbooking_risk import compute_overbooking_proxy_table, lookup_overbooking_risk
from src.recommender import RecommendationRequest, recommend_explainable
from src.data_loader import load_all_data, prepare_hotel_candidates, load_review_summaries, load_hotel_bookings
from src.price_insights import compute_booking_window_table, compute_expected_adr_table, best_booking_window, price_fairness
from src.config import DEFAULT_CONFIG
from src.utils import get_project_root
from src.logger import logger


class RecommendPayload(BaseModel):
    city: str | None = None
    budget: float | None = None
    min_rating: float | None = None
    adults: int | None = None
    children: int | None = None
    meal: str | None = None
    hotel_type: str | None = None
    limit: int = 10
    include_cancellation_risk: bool = False
    market_segment: str | None = None
    lead_time: int | None = None


class ForecastSignalPayload(BaseModel):
    city: str
    hotel_class: float | None = None
    check_in_date: str
    horizon_days: int = 7


class CancellationRiskPayload(BaseModel):
    hotel_type: str
    market_segment: str
    lead_time: int


class OverbookingRiskPayload(BaseModel):
    hotel_type: str
    arrival_month: str
    market_segment: str | None = None
    is_repeated_guest: int | None = None
    previous_cancellations: int | None = None


class PriceFairnessPayload(BaseModel):
    hotel_type: str
    arrival_month: str
    current_price: float
    hotel_class: float | None = None


class BestBookingWindowPayload(BaseModel):
    hotel_type: str
    arrival_month: str
    min_samples: int = 200


app = FastAPI(title="Travel Hotel Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load data at startup
ROOT = get_project_root()
logger.info(f"Loading data from {ROOT}")

try:
    CANDIDATES, REVIEW_SUMMARY, BOOKINGS, CITIES = load_all_data(ROOT)
    CANCELLATION_RISK_TABLE = compute_cancellation_risk_table(BOOKINGS)
    OVERBOOKING_RISK_TABLE = compute_overbooking_proxy_table(BOOKINGS)
    EXPECTED_ADR_TABLE = compute_expected_adr_table(BOOKINGS)
    BOOKING_WINDOW_TABLE = compute_booking_window_table(BOOKINGS)
    logger.info(f"Loaded {len(CANDIDATES)} candidates, {len(REVIEW_SUMMARY)} reviews, {len(BOOKINGS)} bookings")
except Exception as e:
    logger.error(f"Failed to load data: {e}", exc_info=True)
    raise


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta/cities")
def meta_cities() -> dict:
    """Get list of available cities."""
    try:
        if "city" not in CANDIDATES.columns:
            return {"count": 0, "cities": []}
        return {"count": len(CITIES), "cities": CITIES}
    except Exception as e:
        logger.error(f"Error getting cities: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/meta/stats")
def meta_stats() -> dict:
    return {
        "candidates": int(len(CANDIDATES)),
        "review_summaries": int(len(REVIEW_SUMMARY)),
        "has_city": bool("city" in CANDIDATES.columns),
        "has_price_band": bool("adr_low" in CANDIDATES.columns and "adr_high" in CANDIDATES.columns),
    }


@app.post("/recommend")
def recommend(payload: RecommendPayload) -> dict:
    """Get hotel recommendations based on criteria."""
    try:
        logger.info(f"Recommendation request: city={payload.city}, budget={payload.budget}, limit={payload.limit}")
        
        req = RecommendationRequest(
            city=payload.city,
            budget=payload.budget,
            min_rating=payload.min_rating,
            adults=payload.adults,
            children=payload.children,
            meal=payload.meal,
            hotel_type=payload.hotel_type,
        )

        ranked = recommend_explainable(
            candidates=CANDIDATES,
            review_summary=REVIEW_SUMMARY,
            req=req,
            limit=int(payload.limit),
        )

        cancellation_risk = None
        if payload.include_cancellation_risk:
            hotel_type = payload.hotel_type
            market_segment = payload.market_segment
            lead_time = payload.lead_time

            # If hotel_type wasn't provided by the user, fall back to the most common booking "hotel" type.
            if hotel_type is None:
                try:
                    hotel_type = str(BOOKINGS["hotel"].mode().iloc[0])
                except Exception:
                    hotel_type = None

            risk = lookup_cancellation_risk(
                risk_table=CANCELLATION_RISK_TABLE,
                hotel_type=hotel_type,
                market_segment=market_segment,
                lead_time=lead_time,
            )
            cancellation_risk = risk.__dict__ if risk is not None else None

        records = ranked.fillna("").to_dict(orient="records")
        
        logger.info(f"Returning {len(records)} recommendations")

        return {
            "count": len(records),
            "results": records,
            "cancellation_risk": cancellation_risk,
        }
    except Exception as e:
        logger.error(f"Error in recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk/cancellation")
def risk_cancellation(payload: CancellationRiskPayload) -> dict:
    risk = lookup_cancellation_risk(
        risk_table=CANCELLATION_RISK_TABLE,
        hotel_type=payload.hotel_type,
        market_segment=payload.market_segment,
        lead_time=payload.lead_time,
    )
    return {"result": risk.__dict__ if risk is not None else None}


@app.post("/risk/overbooking")
def risk_overbooking(payload: OverbookingRiskPayload) -> dict:
    risk = lookup_overbooking_risk(
        table=OVERBOOKING_RISK_TABLE,
        bookings=BOOKINGS,
        hotel_type=payload.hotel_type,
        arrival_month=payload.arrival_month,
        market_segment=payload.market_segment,
        is_repeated_guest=payload.is_repeated_guest,
        previous_cancellations=payload.previous_cancellations,
    )
    return {"result": risk.__dict__ if risk is not None else None}


@app.post("/advice/price_fairness")
def advice_price_fairness(payload: PriceFairnessPayload) -> dict:
    res = price_fairness(
        expected_table=EXPECTED_ADR_TABLE,
        hotel_type=payload.hotel_type,
        arrival_month=payload.arrival_month,
        current_price=payload.current_price,
        hotel_class=payload.hotel_class,
        class_base=DEFAULT_CONFIG.DEFAULT_CLASS_BASE,
    )
    return {"result": res.__dict__ if res is not None else None}


@app.post("/advice/best_booking_window")
def advice_best_booking_window(payload: BestBookingWindowPayload) -> dict:
    res = best_booking_window(
        window_table=BOOKING_WINDOW_TABLE,
        hotel_type=payload.hotel_type,
        arrival_month=payload.arrival_month,
        min_bucket_n=int(payload.min_samples),
    )
    return {"result": res.__dict__ if res is not None else None}


@app.post("/forecast/signal")
def forecast_signal_api(payload: ForecastSignalPayload) -> dict:
    """Get price forecast signal for booking timing advice."""
    try:
        logger.info(f"Forecast request: city={payload.city}, date={payload.check_in_date}")
        sig = forecast_signal(
            root=ROOT,
            city=payload.city,
            hotel_class=payload.hotel_class,
            check_in_date=payload.check_in_date,
            horizon_days=int(payload.horizon_days),
        )
        return {"result": sig.__dict__ if sig is not None else None}
    except Exception as e:
        logger.error(f"Error in forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
