"""
Build offerings.csv and hotel_review_summaries.csv from 'Hotel Reviews.csv'.

Run from project root:
    python scripts/build_data.py
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
SRC = DATA_DIR / "Hotel Reviews.csv"
OFFERINGS_OUT = DATA_DIR / "offerings.csv"
SUMMARIES_OUT = DATA_DIR / "hotel_review_summaries.csv"


# ---------------------------------------------------------------------------
# Category → hotel class heuristic
# ---------------------------------------------------------------------------

def _infer_hotel_class(categories: str) -> float:
    if not isinstance(categories, str):
        return 3.5
    cats = categories.lower()
    if any(k in cats for k in ["luxury", "five star", "5 star"]):
        return 5.0
    if any(k in cats for k in ["resort", "spa", "boutique"]):
        return 4.0
    if any(k in cats for k in ["motel", "budget", "economy", "inn"]):
        return 2.5
    if any(k in cats for k in ["bed breakfast", "bed & breakfast", "b&b"]):
        return 3.0
    return 3.5


# ---------------------------------------------------------------------------
# Sentiment from rating (simple proxy, no ML required)
# ---------------------------------------------------------------------------

def _rating_to_sentiment(rating: float) -> float:
    """Map 1-5 star rating to 0-1 sentiment score."""
    return float(np.clip((rating - 1) / 4.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Keyword-based pros / cons extraction
# ---------------------------------------------------------------------------

_PRO_MAP = {
    "clean rooms": ["clean", "spotless", "tidy"],
    "friendly staff": ["friendly", "helpful", "staff", "service", "welcoming", "warm"],
    "location": ["location", "close", "convenient", "walk", "central"],
    "comfortable": ["comfortable", "cozy", "spacious", "bed", "beds"],
    "view": ["view", "views", "scenic"],
    "food": ["breakfast", "food", "restaurant", "dining"],
    "good wifi": ["wifi", "wi-fi", "internet"],
    "value": ["value", "affordable", "worth"],
}

_CON_MAP = {
    "slow wifi": ["wifi", "wi-fi", "internet", "slow"],
    "noisy rooms": ["noisy", "noise", "loud"],
    "cleanliness": ["dirty", "smell", "smelly", "bugs", "pest"],
    "expensive": ["expensive", "overpriced", "pricey"],
    "small rooms": ["small", "tiny", "cramped"],
    "broken": ["broken", "maintenance", "old"],
    "rude staff": ["rude", "unfriendly", "unhelpful"],
}

_STOP = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for",
    "with", "is", "it", "this", "that", "was", "were", "are", "be", "as",
    "at", "by", "from", "we", "i", "you", "they", "very", "really", "so",
    "too", "just", "also", "not", "no", "yes", "can", "could", "would",
    "have", "had", "our", "my", "your",
}


def _tokenize(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"[a-zA-Z]{2,}", text.lower())) - _STOP


def _extract_pros_cons(text: str, rating: float) -> tuple[list[str], list[str]]:
    toks = _tokenize(text)
    pros: list[str] = []
    cons: list[str] = []

    if rating >= 3.5:
        for label, keywords in _PRO_MAP.items():
            if any(k in toks for k in keywords):
                pros.append(label)

    if rating <= 3.0:
        for label, keywords in _CON_MAP.items():
            if any(k in toks for k in keywords):
                cons.append(label)
    elif rating <= 4.0:
        for label, keywords in _CON_MAP.items():
            if any(k in toks for k in keywords):
                cons.append(label)

    return pros, cons


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Reading {SRC} ...")
    df = pd.read_csv(SRC)
    print(f"  {len(df):,} rows, {df['id'].nunique():,} unique hotels")

    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    df = df[df["reviews.rating"].notna()].copy()

    # -----------------------------------------------------------------------
    # Build offerings.csv — one row per hotel
    # -----------------------------------------------------------------------
    print("Building offerings.csv ...")
    hotels = (
        df.groupby("id")
        .first()
        .reset_index()[["id", "name", "city", "categories", "sourceURLs", "websites"]]
    )
    hotels["hotel_class"] = hotels["categories"].apply(_infer_hotel_class)
    hotels["type"] = "hotel"
    hotels["url"] = hotels["sourceURLs"].fillna(hotels["websites"]).fillna("")
    hotels["region_id"] = ""
    hotels["address"] = hotels["city"].apply(lambda c: str({"locality": c}))
    hotels = hotels.rename(columns={"id": "id", "name": "name"})[
        ["id", "name", "city", "hotel_class", "region_id", "url", "address", "type"]
    ]

    hotels.to_csv(OFFERINGS_OUT, index=False)
    print(f"  Saved {len(hotels):,} hotels -> {OFFERINGS_OUT}")

    # -----------------------------------------------------------------------
    # Build hotel_review_summaries.csv -- aggregated per hotel
    # -----------------------------------------------------------------------
    print("Building hotel_review_summaries.csv ...")

    rows = []
    grouped = df.groupby("id")
    total = len(grouped)
    for i, (hotel_id, grp) in enumerate(grouped, 1):
        if i % 200 == 0:
            print(f"  Processing {i}/{total} hotels ...")

        hotel_name = grp["name"].iloc[0]
        city = grp["city"].iloc[0]
        ratings = grp["reviews.rating"].dropna()
        avg_rating = float(ratings.mean()) if len(ratings) > 0 else None
        n_reviews = len(grp)

        # Sentiment: average of rating-derived scores
        sentiment_score = float(ratings.apply(_rating_to_sentiment).mean()) if len(ratings) > 0 else None

        # Pros / cons: aggregate across all reviews
        pro_counter: Counter = Counter()
        con_counter: Counter = Counter()
        for _, row in grp.iterrows():
            text = str(row.get("reviews.text", "") or "")
            title = str(row.get("reviews.title", "") or "")
            combined = f"{title} {text}"
            rating = float(row["reviews.rating"]) if pd.notna(row["reviews.rating"]) else 3.0
            pros, cons = _extract_pros_cons(combined, rating)
            pro_counter.update(pros)
            con_counter.update(cons)

        top_pros = [p for p, _ in pro_counter.most_common(6)]
        top_cons = [c for c, _ in con_counter.most_common(6)]

        rows.append({
            "offering_id": hotel_id,
            "hotel": hotel_name,
            "city": city,
            "avg_rating": round(avg_rating, 4) if avg_rating is not None else None,
            "sentiment_score": round(sentiment_score, 4) if sentiment_score is not None else None,
            "pros": top_pros,
            "cons": top_cons,
            "n_reviews": n_reviews,
        })

    summaries = pd.DataFrame(rows)
    summaries.to_csv(SUMMARIES_OUT, index=False)
    print(f"  Saved {len(summaries):,} hotel summaries -> {SUMMARIES_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
