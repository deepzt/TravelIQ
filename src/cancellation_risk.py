from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_LEAD_BUCKETS = [0, 7, 30, 90, 180]  # right edges (inclusive); 181+ is the last


def _lead_bucket(lead_time: int) -> str:
    if lead_time <= 7:
        return "0-7"
    if lead_time <= 30:
        return "8-30"
    if lead_time <= 90:
        return "31-90"
    if lead_time <= 180:
        return "91-180"
    return "181+"


@dataclass
class CancellationRisk:
    hotel_type: str
    market_segment: str
    lead_time_bucket: str
    cancellation_rate: float
    canceled_bookings: int
    total_bookings: int
    advice: str


def compute_cancellation_risk_table(bookings: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute cancellation rates grouped by hotel type, market segment, and lead-time bucket.
    """
    df = bookings.copy()
    df["lead_time_bucket"] = df["lead_time"].apply(
        lambda x: _lead_bucket(int(x)) if pd.notna(x) else "unknown"
    )
    df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce").fillna(0)

    grp = (
        df.groupby(["hotel", "market_segment", "lead_time_bucket"])["is_canceled"]
        .agg(["mean", "sum", "count"])
        .reset_index()
        .rename(columns={"hotel": "hotel_type", "mean": "cancellation_rate",
                         "sum": "canceled_bookings", "count": "total_bookings"})
    )
    return grp


def lookup_cancellation_risk(
    risk_table: pd.DataFrame,
    hotel_type: str | None,
    market_segment: str | None,
    lead_time: int | None,
) -> CancellationRisk | None:
    """
    Find the best-matching row in the pre-computed risk table.
    Falls back to broader matches when exact matches are unavailable.
    """
    if risk_table is None or risk_table.empty:
        return None

    lt_bucket = _lead_bucket(int(lead_time)) if lead_time is not None else None

    df = risk_table.copy()

    # Try progressively looser matches
    for use_segment in [True, False]:
        for use_bucket in [True, False]:
            mask = pd.Series([True] * len(df), index=df.index)
            if hotel_type:
                mask &= df["hotel_type"].str.lower().eq(hotel_type.lower())
            if use_segment and market_segment:
                mask &= df["market_segment"].str.lower().eq(market_segment.lower())
            if use_bucket and lt_bucket:
                mask &= df["lead_time_bucket"].eq(lt_bucket)

            sub = df[mask]
            if not sub.empty:
                row = sub.sort_values("total_bookings", ascending=False).iloc[0]
                rate = float(row["cancellation_rate"])
                advice = _cancellation_advice(rate, market_segment or "", lead_time or 0)
                return CancellationRisk(
                    hotel_type=str(row["hotel_type"]),
                    market_segment=str(row.get("market_segment", market_segment or "")),
                    lead_time_bucket=str(row["lead_time_bucket"]),
                    cancellation_rate=rate,
                    canceled_bookings=int(row["canceled_bookings"]),
                    total_bookings=int(row["total_bookings"]),
                    advice=advice,
                )
    return None


def _cancellation_advice(rate: float, segment: str, lead_time: int) -> str:
    if rate >= 0.40:
        return (
            f"High cancellation risk ({rate:.0%}). Consider a flexible booking policy "
            "or request a deposit to protect your reservation."
        )
    if rate >= 0.25:
        return (
            f"Moderate cancellation risk ({rate:.0%}). Review the hotel's cancellation "
            "policy before confirming."
        )
    return (
        f"Low cancellation risk ({rate:.0%}). This booking profile historically has "
        "good reliability."
    )
