from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OverbookingRisk:
    hotel_type: str
    arrival_month: str
    risk_level: str          # low / medium / high
    risk_score: float        # 0.0 – 1.0
    reassignment_rate: float
    waiting_list_rate: float
    advice: str


def compute_overbooking_proxy_table(bookings: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute a proxy overbooking risk table.
    Risk signals:
      - reassigned: reserved_room_type != assigned_room_type
      - waitlisted: days_in_waiting_list > 0
    """
    df = bookings.copy()

    if "reserved_room_type" in df.columns and "assigned_room_type" in df.columns:
        df["reassigned"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)
    else:
        df["reassigned"] = 0

    if "days_in_waiting_list" in df.columns:
        df["waitlisted"] = (pd.to_numeric(df["days_in_waiting_list"], errors="coerce").fillna(0) > 0).astype(int)
    else:
        df["waitlisted"] = 0

    group_cols = ["hotel", "arrival_date_month"]
    if "market_segment" in df.columns:
        group_cols.append("market_segment")

    grp = (
        df.groupby(group_cols)
        .agg(
            reassignment_rate=("reassigned", "mean"),
            waiting_list_rate=("waitlisted", "mean"),
            total_bookings=("reassigned", "count"),
        )
        .reset_index()
        .rename(columns={"hotel": "hotel_type"})
    )
    return grp


def lookup_overbooking_risk(
    table: pd.DataFrame,
    bookings: pd.DataFrame,
    hotel_type: str,
    arrival_month: str,
    market_segment: str | None = None,
    is_repeated_guest: int | None = None,
    previous_cancellations: int | None = None,
) -> OverbookingRisk | None:
    if table is None or table.empty:
        return None

    df = table.copy()

    # Try exact match first, then fall back
    for use_segment in [True, False]:
        mask = (
            df["hotel_type"].str.lower().eq(hotel_type.lower()) &
            df["arrival_date_month"].str.lower().eq(arrival_month.lower())
        )
        if use_segment and market_segment and "market_segment" in df.columns:
            mask &= df["market_segment"].str.lower().eq(market_segment.lower())

        sub = df[mask]
        if not sub.empty:
            row = sub.sort_values("total_bookings", ascending=False).iloc[0]

            reassign_rate = float(row["reassignment_rate"])
            wait_rate = float(row["waiting_list_rate"])
            risk_score = 0.7 * reassign_rate + 0.3 * wait_rate

            # Adjustments
            if is_repeated_guest:
                risk_score = max(0.0, risk_score - 0.03)
            if previous_cancellations and previous_cancellations > 2:
                risk_score = min(1.0, risk_score + 0.05)

            if risk_score >= 0.25:
                level = "high"
            elif risk_score >= 0.12:
                level = "medium"
            else:
                level = "low"

            return OverbookingRisk(
                hotel_type=hotel_type,
                arrival_month=arrival_month,
                risk_level=level,
                risk_score=round(risk_score, 4),
                reassignment_rate=round(reassign_rate, 4),
                waiting_list_rate=round(wait_rate, 4),
                advice=_overbooking_advice(level, reassign_rate, arrival_month),
            )
    return None


def _overbooking_advice(level: str, reassign_rate: float, month: str) -> str:
    if level == "high":
        return (
            f"High overbooking risk in {month} ({reassign_rate:.0%} room reassignment rate). "
            "Contact the hotel to confirm your specific room type before arrival."
        )
    if level == "medium":
        return (
            f"Moderate overbooking risk in {month}. "
            "You may want to reconfirm your booking closer to arrival."
        )
    return f"Low overbooking risk in {month}. Booking looks solid based on historical patterns."
