from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DEFAULT_FEATURE_COLS: list[str] = [
    "hotel",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status_date",
]


def prepare_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter and clean the bookings frame for model training."""
    out = df.copy()
    out["adr"] = pd.to_numeric(out["adr"], errors="coerce")
    out = out[out["adr"].notna() & (out["adr"] >= 0)]
    q99 = out["adr"].quantile(0.99)
    out = out[out["adr"] <= q99]
    return out.reset_index(drop=True)


def train_price_model(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, float]:
    """
    Train a HistGradientBoosting price model.

    Returns:
        pipe – fitted sklearn Pipeline
        mae  – mean absolute error on the hold-out test set
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    frame = prepare_training_frame(df)
    available = [c for c in feature_cols if c in frame.columns]
    X = frame[available]
    y = frame["adr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    cat_cols = [c for c in available if X_train[c].dtype == "object"]
    num_cols = [c for c in available if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                cat_cols,
            ),
        ]
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", HistGradientBoostingRegressor(random_state=random_state)),
    ])
    pipe.fit(X_train, y_train)
    mae = float(mean_absolute_error(y_test, pipe.predict(X_test)))
    return pipe, mae


@dataclass
class PriceTimingSignal:
    adr_now: float
    adr_if_wait: float
    delta: float
    pct_change: float
    decision: str   # BOOK_NOW / WAIT / STABLE
    wait_days: int
    lead_time_now: int
    lead_time_if_wait: int


def price_timing_signal(
    pipe: Pipeline,
    sample: dict,
    wait_days: int = 7,
    feature_cols: list[str] | None = None,
) -> PriceTimingSignal:
    """
    Compare predicted ADR now vs after waiting `wait_days`.
    Waiting reduces lead_time (you book later = fewer days ahead).
    """
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    available = [c for c in feature_cols if c in sample]

    now = {k: sample[k] for k in available}
    later = {k: sample[k] for k in available}

    lt = int(sample.get("lead_time") or 0)
    later["lead_time"] = max(0, lt - wait_days)

    def _predict(row: dict) -> float:
        df = pd.DataFrame([row])
        for col in available:
            if col not in df.columns:
                df[col] = np.nan
        return float(pipe.predict(df[available])[0])

    adr_now = _predict(now)
    adr_later = _predict(later)
    delta = adr_later - adr_now
    pct = (delta / max(1e-6, adr_now)) * 100.0

    if pct > 2.0:
        decision = "BOOK_NOW"
    elif pct < -2.0:
        decision = "WAIT"
    else:
        decision = "STABLE"

    return PriceTimingSignal(
        adr_now=round(adr_now, 2),
        adr_if_wait=round(adr_later, 2),
        delta=round(delta, 2),
        pct_change=round(pct, 2),
        decision=decision,
        wait_days=wait_days,
        lead_time_now=lt,
        lead_time_if_wait=later["lead_time"],
    )
