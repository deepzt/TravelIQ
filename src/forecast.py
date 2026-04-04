from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_hotel_bookings
from src.utils import parse_locality, get_project_root
from src.config import DEFAULT_CONFIG


@dataclass
class ForecastSignal:
    city: str
    check_in_date: str
    trend: str                   # increasing / decreasing / stable
    expected_change: str         # e.g. "+3.2%"
    confidence: float            # 0.0 – 1.0
    hotel_price_volatility: str  # low / medium / high
    booking_advice: str          # BOOK_NOW / WAIT / WATCH / STABLE


# City and class multipliers make the signal context-aware
_CITY_MULT: dict[str, float] = {
    "new york city": 1.45,
    "los angeles": 1.30,
    "san francisco": 1.35,
    "chicago": 1.15,
    "miami": 1.20,
    "las vegas": 1.10,
    "london": 1.40,
    "paris": 1.38,
    "tokyo": 1.25,
    "dubai": 1.30,
}
_DEFAULT_CITY_MULT = 1.0


def _city_multiplier(city: str) -> float:
    return _CITY_MULT.get(city.strip().lower(), _DEFAULT_CITY_MULT)


def _class_multiplier(hotel_class: float | None, class_base: float = 3.5) -> float:
    if hotel_class is None or hotel_class <= 0:
        return 1.0
    return float(np.clip(hotel_class / class_base, 0.6, 2.0))


def _fit_linear_trend(values: np.ndarray) -> tuple[float, float]:
    """Fit a simple linear trend. Returns (slope, r_squared)."""
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    y_mean = values.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom < 1e-10:
        return 0.0, 0.0
    slope = float(((x - x_mean) * (values - y_mean)).sum() / denom)
    y_hat = x_mean * slope + y_mean + slope * (x - x_mean)
    ss_res = ((values - y_hat) ** 2).sum()
    ss_tot = ((values - y_mean) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    return slope, float(np.clip(r2, 0.0, 1.0))


def _volatility_label(cv: float) -> str:
    """Coefficient of variation → volatility bucket."""
    if cv < 0.15:
        return "low"
    if cv < 0.35:
        return "medium"
    return "high"


def forecast_signal(
    root: Path | None = None,
    city: str = "",
    hotel_class: float | None = None,
    check_in_date: str = "",
    horizon_days: int = 7,
) -> ForecastSignal | None:
    """
    Produce a price-direction signal from historical ADR trends.

    Strategy:
    1. Load bookings; build a monthly ADR time series.
    2. Fit a linear trend.
    3. Scale the forecast by city and hotel-class multipliers.
    4. Map to BOOK_NOW / WAIT / WATCH / STABLE advice.
    """
    if root is None:
        root = get_project_root()

    try:
        bookings = load_hotel_bookings(root)
    except FileNotFoundError:
        return None

    # Build monthly time series
    df = bookings.copy()
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df = df[df["adr"].notna() & (df["adr"] > 0)]

    month_order = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }

    if "arrival_date_month" in df.columns and "arrival_date_year" in df.columns:
        df["_month_num"] = df["arrival_date_month"].str.lower().map(month_order)
        df["_period"] = df["arrival_date_year"].astype(str) + "-" + df["_month_num"].astype(str).str.zfill(2)
        ts = df.groupby("_period")["adr"].median().sort_index()
    else:
        return None

    if len(ts) < 3:
        return None

    values = ts.values.astype(float)
    slope, r2 = _fit_linear_trend(values)

    # Context-aware scaling
    city_mult = _city_multiplier(city)
    class_mult = _class_multiplier(hotel_class, DEFAULT_CONFIG.DEFAULT_CLASS_BASE)
    combined_mult = city_mult * class_mult

    # Extrapolate horizon
    last_val = float(values[-1]) * combined_mult
    monthly_change = slope * (horizon_days / 30.0) * combined_mult
    pct_change = (monthly_change / max(1.0, last_val)) * 100.0

    # Coefficient of variation for volatility
    cv = float(np.std(values) / max(1e-6, np.mean(values)))
    volatility = _volatility_label(cv)

    # Trend label
    if pct_change > 1.5:
        trend = "increasing"
    elif pct_change < -1.5:
        trend = "decreasing"
    else:
        trend = "stable"

    # Confidence: r² weighted by sample size (more data = more trustworthy)
    n_periods = len(values)
    size_factor = float(np.clip(n_periods / 24, 0.3, 1.0))  # 24 months ≈ full confidence
    confidence = float(np.clip(r2 * size_factor * (1.0 - 0.5 * cv), 0.05, 0.95))

    # Booking advice
    if trend == "increasing" and confidence > 0.4:
        advice = "BOOK_NOW"
    elif trend == "decreasing" and confidence > 0.4:
        advice = "WAIT"
    elif volatility == "high":
        advice = "WATCH"
    else:
        advice = "STABLE"

    sign = "+" if pct_change >= 0 else ""
    expected_change = f"{sign}{pct_change:.1f}%"

    return ForecastSignal(
        city=city,
        check_in_date=check_in_date,
        trend=trend,
        expected_change=expected_change,
        confidence=round(confidence, 3),
        hotel_price_volatility=volatility,
        booking_advice=advice,
    )
