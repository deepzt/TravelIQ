from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import parse_locality, get_project_root
from src.config import DEFAULT_CONFIG
from src.logger import logger


def _find_bookings_csv(root: Path) -> Path:
    """hotel_bookings.csv may live at root or root/data/."""
    candidates = [
        root / "hotel_bookings.csv",
        root / "data" / "hotel_bookings.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"hotel_bookings.csv not found. Tried: {[str(c) for c in candidates]}"
    )


def load_hotel_bookings(root: Path | None = None) -> pd.DataFrame:
    """Load and minimally clean the hotel bookings dataset."""
    if root is None:
        root = get_project_root()
    path = _find_bookings_csv(root)
    logger.info(f"Loading bookings from {path}")
    df = pd.read_csv(path)
    df["adr"] = pd.to_numeric(df["adr"], errors="coerce")
    df = df[df["adr"].notna() & (df["adr"] >= 0)].copy()
    return df


def load_offerings(root: Path | None = None) -> pd.DataFrame:
    """Load the offerings (hotel metadata) dataset."""
    if root is None:
        root = get_project_root()
    path = root / "data" / "offerings.csv"
    if not path.exists():
        logger.warning(f"offerings.csv not found at {path}. Hotel recommendations will be unavailable.")
        return pd.DataFrame(columns=["id", "name", "hotel_class", "region_id", "url", "address", "type"])
    logger.info(f"Loading offerings from {path}")
    return pd.read_csv(path)


def load_review_summaries(root: Path | None = None) -> pd.DataFrame:
    """Load pre-computed hotel review summaries."""
    if root is None:
        root = get_project_root()
    path = root / "data" / "hotel_review_summaries.csv"
    if not path.exists():
        logger.warning(f"Review summaries not found at {path}. Returning empty frame.")
        return pd.DataFrame(columns=["offering_id", "hotel", "city", "avg_rating",
                                     "sentiment_score", "pros", "cons", "n_reviews"])
    logger.info(f"Loading review summaries from {path}")
    df = pd.read_csv(path)
    # Ensure offering_id is string to match offerings dataset
    if "offering_id" in df.columns:
        df["offering_id"] = df["offering_id"].astype(str)
    return df


def prepare_hotel_candidates(
    offerings: pd.DataFrame,
    bookings: pd.DataFrame,
    class_base: float | None = None,
) -> pd.DataFrame:
    """
    Build the hotel candidate pool from offerings, enriched with estimated ADR.
    Price estimates come from a global ADR baseline scaled by hotel class and city.
    """
    if class_base is None:
        class_base = DEFAULT_CONFIG.DEFAULT_CLASS_BASE

    hotels = offerings[offerings["type"].eq("hotel")].copy()
    hotels = hotels.rename(columns={"id": "offering_id", "name": "hotel"})
    # Use city column directly if available, otherwise parse from address dict
    if "city" in hotels.columns:
        hotels["city"] = hotels["city"].where(hotels["city"].notna(), None)
    else:
        hotels["city"] = hotels["address"].apply(parse_locality)
    hotels["hotel_class"] = pd.to_numeric(hotels["hotel_class"], errors="coerce")

    # Global ADR baseline from bookings
    base_adr = float(bookings["adr"].median())

    # Class multiplier
    hotels["_class_mult"] = (
        hotels["hotel_class"].fillna(class_base) / class_base
    ).clip(0.6, 2.0)

    # City multiplier: relative average hotel class per city vs global mean
    city_class_mean = hotels.groupby("city")["hotel_class"].mean()
    global_class_mean = float(hotels["hotel_class"].mean()) if hotels["hotel_class"].notna().any() else class_base
    hotels["_city_mult"] = (
        hotels["city"]
        .map(city_class_mean / max(1e-6, global_class_mean))
        .fillna(1.0)
        .clip(0.7, 1.6)
    )

    hotels["adr_est"] = (base_adr * hotels["_class_mult"] * hotels["_city_mult"]).round(2)
    hotels["adr_low"] = (hotels["adr_est"] * 0.85).round(2)
    hotels["adr_high"] = (hotels["adr_est"] * 1.20).round(2)
    hotels["adr"] = hotels["adr_est"]
    hotels["price_confidence"] = np.where(
        hotels["city"].notna() & hotels["hotel_class"].notna(), "medium", "low"
    )

    # Ensure offering_id is string so it merges correctly with review summaries
    if "offering_id" in hotels.columns:
        hotels["offering_id"] = hotels["offering_id"].astype(str)

    keep_cols = [
        c for c in [
            "offering_id", "hotel", "city", "hotel_class", "region_id",
            "url", "adr", "adr_low", "adr_high", "price_confidence",
        ]
        if c in hotels.columns
    ]
    return hotels[keep_cols].reset_index(drop=True)


def load_all_data(
    root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load all datasets in one call.

    Returns:
        candidates     – hotel candidate pool (offerings enriched with price estimates)
        review_summary – hotel-level review aggregates
        bookings       – raw hotel bookings (filtered)
        cities         – sorted list of available city names
    """
    if root is None:
        root = get_project_root()

    bookings = load_hotel_bookings(root)
    offerings = load_offerings(root)
    review_summary = load_review_summaries(root)
    candidates = prepare_hotel_candidates(offerings, bookings)

    cities: list[str] = []
    if "city" in candidates.columns:
        cities = sorted(candidates["city"].dropna().unique().tolist())

    logger.info(
        f"load_all_data: {len(candidates)} candidates, "
        f"{len(review_summary)} reviews, {len(bookings)} bookings, "
        f"{len(cities)} cities"
    )
    return candidates, review_summary, bookings, cities
