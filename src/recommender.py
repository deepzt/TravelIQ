from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RecommendationRequest:
    city: str | None = None
    budget: float | None = None
    min_rating: float | None = None
    adults: int | None = None
    children: int | None = None
    meal: str | None = None
    hotel_type: str | None = None


# Scoring weights (must sum to 1.0)
_W_PRICE = 0.30
_W_SENTIMENT = 0.30
_W_RATING = 0.25
_W_VOLUME = 0.15


def _price_score(adr_low: float, adr_high: float, budget: float) -> float:
    """
    1.0 if budget comfortably covers the low estimate.
    Linearly decays as price exceeds budget.
    """
    if budget <= 0:
        return 0.5
    if adr_low <= budget:
        # Reward being well within budget
        return float(np.clip(1.0 - (adr_low / budget - 0.5), 0.5, 1.0))
    # Price exceeds budget: penalise proportionally
    excess = (adr_low - budget) / max(1.0, budget)
    return float(np.clip(1.0 - excess, 0.0, 0.5))


def _volume_score(n_reviews: float) -> float:
    """Log-normalised review volume score, capped at 5000 reviews."""
    if n_reviews <= 0:
        return 0.0
    return float(np.clip(np.log1p(n_reviews) / np.log1p(5000), 0.0, 1.0))


def _build_reason(row: pd.Series) -> str:
    parts: list[str] = []
    sent = row.get("sentiment_score")
    if sent is not None and not (isinstance(sent, float) and np.isnan(sent)):
        parts.append(f"{float(sent):.0%} positive reviews")
    rating = row.get("avg_rating")
    if rating is not None and not (isinstance(rating, float) and np.isnan(rating)):
        parts.append(f"rated {float(rating):.1f}/5")
    adr = row.get("adr")
    if adr is not None and not (isinstance(adr, float) and np.isnan(adr)):
        parts.append(f"~${float(adr):.0f}/night")
    return "Recommended because " + " • ".join(parts) if parts else "Recommended hotel"


def _safe_list(val) -> list:
    """Coerce a stored list string (or actual list) to a Python list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str) and val.startswith("["):
        try:
            return ast.literal_eval(val)
        except Exception:
            pass
    return []


def recommend_explainable(
    candidates: pd.DataFrame,
    review_summary: pd.DataFrame,
    req: RecommendationRequest,
    limit: int = 10,
) -> pd.DataFrame:
    """
    Return a ranked DataFrame of hotel recommendations.

    Hard filters applied first (city, budget, min_rating).
    Weighted score computed from price fit, sentiment, rating, review volume.
    """
    pool = candidates.copy()

    # --- Hard filters ---
    if req.city:
        city_lower = req.city.strip().lower()
        if "city" in pool.columns:
            pool = pool[pool["city"].str.lower().eq(city_lower)]

    # Merge review signals
    if not review_summary.empty and "offering_id" in pool.columns and "offering_id" in review_summary.columns:
        rev_cols = [c for c in ["offering_id", "avg_rating", "sentiment_score", "n_reviews", "pros", "cons"]
                    if c in review_summary.columns]
        pool = pool.merge(review_summary[rev_cols], on="offering_id", how="left")

    # Ensure numeric columns exist
    for col in ["avg_rating", "sentiment_score", "n_reviews", "adr_low", "adr_high", "adr"]:
        if col not in pool.columns:
            pool[col] = np.nan
        pool[col] = pd.to_numeric(pool[col], errors="coerce")

    # Filter by minimum rating (after merge so we have avg_rating)
    if req.min_rating is not None and req.min_rating > 0:
        # Keep hotels without ratings (unknown is not disqualified)
        pool = pool[pool["avg_rating"].isna() | (pool["avg_rating"] >= req.min_rating)]

    if pool.empty:
        return pd.DataFrame()

    # --- Scoring ---
    budget = req.budget or 0.0

    def _score_row(r: pd.Series) -> float:
        # Price component
        adr_low = r.get("adr_low") or r.get("adr") or 0.0
        adr_high = r.get("adr_high") or adr_low
        if np.isnan(adr_low):
            adr_low = 0.0
        if np.isnan(adr_high):
            adr_high = adr_low

        if budget > 0:
            p_score = _price_score(float(adr_low), float(adr_high), budget)
        else:
            p_score = 0.5  # neutral when no budget specified

        # Sentiment component
        sent = r.get("sentiment_score")
        s_score = float(sent) if sent is not None and not (isinstance(sent, float) and np.isnan(sent)) else 0.5

        # Rating component
        rating = r.get("avg_rating")
        r_score = (float(rating) / 5.0) if rating is not None and not (isinstance(rating, float) and np.isnan(rating)) else 0.5

        # Volume component
        n = r.get("n_reviews")
        v_score = _volume_score(float(n)) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0.0

        return _W_PRICE * p_score + _W_SENTIMENT * s_score + _W_RATING * r_score + _W_VOLUME * v_score

    pool["score"] = pool.apply(_score_row, axis=1)
    pool = pool.sort_values("score", ascending=False).head(limit).reset_index(drop=True)

    # Build reason strings
    pool["reason"] = pool.apply(_build_reason, axis=1)

    # Normalise pros/cons to lists
    if "pros" in pool.columns:
        pool["pros"] = pool["pros"].apply(_safe_list)
    if "cons" in pool.columns:
        pool["cons"] = pool["cons"].apply(_safe_list)

    return pool
