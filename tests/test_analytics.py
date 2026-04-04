"""
Unit tests for TravelIQ analytics modules.

Run with:  pytest tests/test_analytics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cancellation_risk import (
    compute_cancellation_risk_table,
    lookup_cancellation_risk,
    _lead_bucket,
)
from src.overbooking_risk import (
    compute_overbooking_proxy_table,
    lookup_overbooking_risk,
)
from src.price_insights import (
    compute_expected_adr_table,
    compute_booking_window_table,
    price_fairness,
    best_booking_window,
)
from src.recommender import (
    RecommendationRequest,
    recommend_explainable,
    _price_score,
    _volume_score,
)
from src.utils import parse_locality, safe_numeric


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bookings(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
    segments = ["Online TA", "Direct", "Corporate", "Groups"]
    hotels = ["City Hotel", "Resort Hotel"]
    room_types = ["A", "B", "C", "D"]

    return pd.DataFrame({
        "hotel": rng.choice(hotels, n),
        "is_canceled": rng.integers(0, 2, n),
        "lead_time": rng.integers(0, 365, n),
        "arrival_date_month": rng.choice(months, n),
        "arrival_date_year": rng.choice([2015, 2016, 2017], n),
        "market_segment": rng.choice(segments, n),
        "adr": rng.uniform(50, 300, n),
        "reserved_room_type": rng.choice(room_types, n),
        "assigned_room_type": rng.choice(room_types, n),
        "days_in_waiting_list": rng.integers(0, 10, n),
    })


def _make_candidates(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    cities = ["New York City", "Los Angeles", "Chicago"]
    return pd.DataFrame({
        "offering_id": range(n),
        "hotel": [f"Hotel {i}" for i in range(n)],
        "city": rng.choice(cities, n),
        "hotel_class": rng.uniform(2.0, 5.0, n).round(1),
        "adr": rng.uniform(80, 250, n).round(2),
        "adr_low": rng.uniform(60, 200, n).round(2),
        "adr_high": rng.uniform(200, 350, n).round(2),
        "price_confidence": rng.choice(["low", "medium"], n),
    })


def _make_reviews(offering_ids) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = len(offering_ids)
    return pd.DataFrame({
        "offering_id": offering_ids,
        "avg_rating": rng.uniform(3.0, 5.0, n).round(2),
        "sentiment_score": rng.uniform(0.4, 1.0, n).round(4),
        "n_reviews": rng.integers(50, 2000, n),
        "pros": [["clean rooms", "location"]] * n,
        "cons": [["slow wifi"]] * n,
    })


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

class TestParseLocality:
    def test_valid_dict(self):
        assert parse_locality("{'locality': 'New York City', 'country': 'US'}") == "New York City"

    def test_missing_locality(self):
        assert parse_locality("{'country': 'US'}") is None

    def test_none(self):
        assert parse_locality(None) is None

    def test_plain_string(self):
        assert parse_locality("not a dict") is None


class TestSafeNumeric:
    def test_int(self):
        assert safe_numeric(42) == 42.0

    def test_float_string(self):
        assert safe_numeric("3.14") == pytest.approx(3.14)

    def test_invalid(self):
        assert safe_numeric("abc") is None

    def test_default(self):
        assert safe_numeric("bad", default=0.0) == 0.0


# ---------------------------------------------------------------------------
# Cancellation Risk
# ---------------------------------------------------------------------------

class TestCancellationRisk:
    def setup_method(self):
        self.bookings = _make_bookings(500)
        self.table = compute_cancellation_risk_table(self.bookings)

    def test_table_not_empty(self):
        assert len(self.table) > 0

    def test_table_columns(self):
        expected = {"hotel_type", "market_segment", "lead_time_bucket",
                    "cancellation_rate", "canceled_bookings", "total_bookings"}
        assert expected.issubset(set(self.table.columns))

    def test_rate_in_range(self):
        assert (self.table["cancellation_rate"] >= 0).all()
        assert (self.table["cancellation_rate"] <= 1).all()

    def test_lookup_returns_result(self):
        result = lookup_cancellation_risk(self.table, "City Hotel", "Online TA", 30)
        assert result is not None
        assert 0.0 <= result.cancellation_rate <= 1.0
        assert result.total_bookings > 0

    def test_lookup_unknown_segment_falls_back(self):
        result = lookup_cancellation_risk(self.table, "City Hotel", "NonExistentSegment", 30)
        assert result is not None

    def test_lookup_empty_table(self):
        result = lookup_cancellation_risk(pd.DataFrame(), "City Hotel", "Direct", 10)
        assert result is None

    def test_lead_bucket_boundaries(self):
        assert _lead_bucket(0) == "0-7"
        assert _lead_bucket(7) == "0-7"
        assert _lead_bucket(8) == "8-30"
        assert _lead_bucket(30) == "8-30"
        assert _lead_bucket(31) == "31-90"
        assert _lead_bucket(181) == "181+"


# ---------------------------------------------------------------------------
# Overbooking Risk
# ---------------------------------------------------------------------------

class TestOverbookingRisk:
    def setup_method(self):
        self.bookings = _make_bookings(500)
        self.table = compute_overbooking_proxy_table(self.bookings)

    def test_table_not_empty(self):
        assert len(self.table) > 0

    def test_rates_in_range(self):
        assert (self.table["reassignment_rate"] >= 0).all()
        assert (self.table["reassignment_rate"] <= 1).all()
        assert (self.table["waiting_list_rate"] >= 0).all()

    def test_lookup_returns_result(self):
        result = lookup_overbooking_risk(
            self.table, self.bookings,
            hotel_type="City Hotel", arrival_month="January",
        )
        assert result is not None
        assert result.risk_level in ("low", "medium", "high")
        assert 0.0 <= result.risk_score <= 1.0

    def test_repeated_guest_lowers_risk(self):
        r_normal = lookup_overbooking_risk(
            self.table, self.bookings, "City Hotel", "January",
            is_repeated_guest=0, previous_cancellations=0,
        )
        r_repeat = lookup_overbooking_risk(
            self.table, self.bookings, "City Hotel", "January",
            is_repeated_guest=1, previous_cancellations=0,
        )
        if r_normal and r_repeat:
            assert r_repeat.risk_score <= r_normal.risk_score

    def test_many_cancellations_raises_risk(self):
        r_normal = lookup_overbooking_risk(
            self.table, self.bookings, "City Hotel", "January",
            is_repeated_guest=0, previous_cancellations=0,
        )
        r_risky = lookup_overbooking_risk(
            self.table, self.bookings, "City Hotel", "January",
            is_repeated_guest=0, previous_cancellations=5,
        )
        if r_normal and r_risky:
            assert r_risky.risk_score >= r_normal.risk_score


# ---------------------------------------------------------------------------
# Price Insights
# ---------------------------------------------------------------------------

class TestPriceFairness:
    def setup_method(self):
        self.bookings = _make_bookings(500)
        self.table = compute_expected_adr_table(self.bookings)

    def test_table_not_empty(self):
        assert len(self.table) > 0

    def test_fair_price(self):
        row = self.table.iloc[0]
        result = price_fairness(
            self.table,
            hotel_type=row["hotel_type"],
            arrival_month=row["arrival_date_month"],
            current_price=float(row["expected_adr"]),
        )
        assert result is not None
        assert result.label in ("Fair price", "Great deal")

    def test_overpriced(self):
        row = self.table.iloc[0]
        result = price_fairness(
            self.table,
            hotel_type=row["hotel_type"],
            arrival_month=row["arrival_date_month"],
            current_price=float(row["expected_adr"]) * 2.0,
        )
        assert result is not None
        assert result.label == "Overpriced"

    def test_great_deal(self):
        row = self.table.iloc[0]
        result = price_fairness(
            self.table,
            hotel_type=row["hotel_type"],
            arrival_month=row["arrival_date_month"],
            current_price=float(row["expected_adr"]) * 0.5,
        )
        assert result is not None
        assert result.label == "Great deal"

    def test_empty_table(self):
        result = price_fairness(pd.DataFrame(), "City Hotel", "January", 100.0)
        assert result is None


class TestBestBookingWindow:
    def setup_method(self):
        self.bookings = _make_bookings(1000)
        self.table = compute_booking_window_table(self.bookings)

    def test_table_not_empty(self):
        assert len(self.table) > 0

    def test_lookup_returns_result(self):
        row = self.table.iloc[0]
        result = best_booking_window(
            self.table,
            hotel_type=row["hotel_type"],
            arrival_month=row["arrival_date_month"],
            min_bucket_n=1,
        )
        assert result is not None
        assert result.min_median_adr > 0
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_table(self):
        result = best_booking_window(pd.DataFrame(), "City Hotel", "January")
        assert result is None


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TestRecommender:
    def test_price_score_within_budget(self):
        score = _price_score(80.0, 120.0, 200.0)
        assert score > 0.5

    def test_price_score_over_budget(self):
        score = _price_score(300.0, 400.0, 100.0)
        assert score < 0.5

    def test_volume_score_zero_reviews(self):
        assert _volume_score(0) == 0.0

    def test_volume_score_increases(self):
        assert _volume_score(100) < _volume_score(1000)

    def test_recommend_returns_dataframe(self):
        candidates = _make_candidates(20)
        reviews = _make_reviews(list(range(20)))
        req = RecommendationRequest(city="New York City", budget=300.0, min_rating=3.0)
        result = recommend_explainable(candidates, reviews, req, limit=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_recommend_respects_limit(self):
        candidates = _make_candidates(20)
        reviews = _make_reviews(list(range(20)))
        req = RecommendationRequest()
        result = recommend_explainable(candidates, reviews, req, limit=3)
        assert len(result) <= 3

    def test_recommend_has_score_and_reason(self):
        candidates = _make_candidates(10)
        reviews = _make_reviews(list(range(10)))
        req = RecommendationRequest()
        result = recommend_explainable(candidates, reviews, req, limit=10)
        assert "score" in result.columns
        assert "reason" in result.columns

    def test_recommend_sorted_by_score(self):
        candidates = _make_candidates(20)
        reviews = _make_reviews(list(range(20)))
        req = RecommendationRequest()
        result = recommend_explainable(candidates, reviews, req, limit=20)
        scores = result["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_recommend_empty_when_no_city_match(self):
        candidates = _make_candidates(10)
        reviews = _make_reviews(list(range(10)))
        req = RecommendationRequest(city="Nonexistent City XYZ")
        result = recommend_explainable(candidates, reviews, req, limit=10)
        assert len(result) == 0
