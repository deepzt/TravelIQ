from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


_LEAD_BUCKET_EDGES = [0, 7, 14, 21, 28, 60, 90, 120, 180]


def _lead_bucket_label(lead_time: int) -> str:
    edges = _LEAD_BUCKET_EDGES
    for i, edge in enumerate(edges):
        if lead_time <= edge:
            if i == 0:
                return f"0-{edge}"
            return f"{edges[i-1]+1}-{edge}"
    return f"{edges[-1]+1}+"


# ---------------------------------------------------------------------------
# Expected ADR table
# ---------------------------------------------------------------------------

def compute_expected_adr_table(bookings: pd.DataFrame) -> pd.DataFrame:
    """Group by hotel type + arrival month and compute median ADR."""
    df = bookings.copy()
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df = df[df["adr"].notna() & (df["adr"] > 0)]

    grp = (
        df.groupby(["hotel", "arrival_date_month"])["adr"]
        .agg(expected_adr="median", n="count")
        .reset_index()
        .rename(columns={"hotel": "hotel_type"})
    )
    return grp


# ---------------------------------------------------------------------------
# Booking window table
# ---------------------------------------------------------------------------

def compute_booking_window_table(bookings: pd.DataFrame) -> pd.DataFrame:
    """Group by hotel type + arrival month + lead-time bucket and compute median ADR."""
    df = bookings.copy()
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df["lead_time"] = pd.to_numeric(df["lead_time"], errors="coerce")
    df = df[df["adr"].notna() & (df["adr"] > 0) & df["lead_time"].notna()]

    df["lead_time_bucket"] = df["lead_time"].apply(lambda x: _lead_bucket_label(int(x)))

    grp = (
        df.groupby(["hotel", "arrival_date_month", "lead_time_bucket"])["adr"]
        .agg(median_adr="median", n="count")
        .reset_index()
        .rename(columns={"hotel": "hotel_type"})
    )
    return grp


# ---------------------------------------------------------------------------
# Price Fairness
# ---------------------------------------------------------------------------

@dataclass
class PriceFairness:
    hotel_type: str
    arrival_month: str
    current_price: float
    expected_price: float
    pct_diff: float     # positive = above market, negative = below
    label: str          # "Great deal" / "Fair price" / "Slightly high" / "Overpriced"
    color: str          # green / yellow / red
    message: str


def price_fairness(
    expected_table: pd.DataFrame,
    hotel_type: str,
    arrival_month: str,
    current_price: float,
    hotel_class: float | None = None,
    class_base: float = 3.5,
) -> PriceFairness | None:
    if expected_table is None or expected_table.empty:
        return None

    # Find best matching row
    mask = (
        expected_table["hotel_type"].str.lower().eq(hotel_type.lower()) &
        expected_table["arrival_date_month"].str.lower().eq(arrival_month.lower())
    )
    sub = expected_table[mask]
    if sub.empty:
        # Fall back: any month for this hotel type
        sub = expected_table[expected_table["hotel_type"].str.lower().eq(hotel_type.lower())]
    if sub.empty:
        return None

    row = sub.sort_values("n", ascending=False).iloc[0]
    expected_adr = float(row["expected_adr"])

    # Adjust for hotel class
    if hotel_class is not None and hotel_class > 0:
        class_mult = float(np.clip(hotel_class / max(1e-6, class_base), 0.6, 2.0))
        expected_adj = expected_adr * class_mult
    else:
        expected_adj = expected_adr

    ratio = current_price / max(1e-6, expected_adj)
    pct_diff = (ratio - 1.0) * 100.0

    if ratio <= 0.92:
        label, color = "Great deal", "green"
        message = f"This price is {abs(pct_diff):.1f}% below the typical market rate. Good time to book."
    elif ratio <= 1.05:
        label, color = "Fair price", "green"
        message = f"This price is in line with the typical market rate (within {abs(pct_diff):.1f}%)."
    elif ratio <= 1.15:
        label, color = "Slightly high", "yellow"
        message = f"This price is {pct_diff:.1f}% above the typical market rate. Consider comparing alternatives."
    else:
        label, color = "Overpriced", "red"
        message = f"This price is {pct_diff:.1f}% above the typical market rate. Strong recommendation to compare."

    return PriceFairness(
        hotel_type=hotel_type,
        arrival_month=arrival_month,
        current_price=round(current_price, 2),
        expected_price=round(expected_adj, 2),
        pct_diff=round(pct_diff, 2),
        label=label,
        color=color,
        message=message,
    )


# ---------------------------------------------------------------------------
# Best Booking Window
# ---------------------------------------------------------------------------

@dataclass
class BookingWindow:
    hotel_type: str
    arrival_month: str
    recommended_window_days: str  # e.g. "22-28"
    min_median_adr: float
    confidence: float             # 0.0 – 1.0
    message: str


def best_booking_window(
    window_table: pd.DataFrame,
    hotel_type: str,
    arrival_month: str,
    min_bucket_n: int = 200,
) -> BookingWindow | None:
    if window_table is None or window_table.empty:
        return None

    mask = (
        window_table["hotel_type"].str.lower().eq(hotel_type.lower()) &
        window_table["arrival_date_month"].str.lower().eq(arrival_month.lower())
    )
    sub = window_table[mask]

    # Relax sample size requirement if too restrictive
    sub_filtered = sub[sub["n"] >= min_bucket_n]
    if sub_filtered.empty and not sub.empty:
        sub_filtered = sub[sub["n"] >= max(1, min_bucket_n // 4)]
    if sub_filtered.empty:
        return None

    best_row = sub_filtered.sort_values("median_adr").iloc[0]
    overall_median = float(sub_filtered["median_adr"].median())
    min_adr = float(best_row["median_adr"])
    bucket = str(best_row["lead_time_bucket"])
    n = int(best_row["n"])

    # Confidence: combination of sample size strength and price separation
    size_score = float(np.clip(np.log1p(n) / np.log1p(5000), 0.0, 1.0))
    separation = max(0.0, (overall_median - min_adr) / max(1e-6, overall_median))
    confidence = float(np.clip(0.5 * size_score + 0.5 * separation, 0.0, 1.0))

    message = (
        f"Book {bucket} days in advance for {arrival_month} — "
        f"median price ~${min_adr:.0f}/night, "
        f"{separation:.0%} below the overall monthly average."
    )

    return BookingWindow(
        hotel_type=hotel_type,
        arrival_month=arrival_month,
        recommended_window_days=bucket,
        min_median_adr=round(min_adr, 2),
        confidence=round(confidence, 4),
        message=message,
    )
