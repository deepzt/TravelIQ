from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st

from src.data_loader import load_all_data, load_hotel_bookings
from src.utils import get_project_root
from src.logger import logger


st.set_page_config(page_title="Hotel Optimizer", layout="wide", initial_sidebar_state="expanded")

root = get_project_root()


def _post_json(base_url: str, path: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + path
    req = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as e:
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        raise RuntimeError(f"HTTP {e.code} calling {url}: {body or e.reason}")
    except URLError as e:
        raise RuntimeError(f"Network error calling {url}: {e.reason}")

# Load data with caching
@st.cache_data(show_spinner=False)
def load_data_cached(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load data with caching."""
    try:
        candidates, review_summary, _, cities = load_all_data(project_root)
        return candidates, review_summary, cities
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        st.error(f"Error loading data: {e}")
        raise


@st.cache_data(show_spinner=False)
def load_bookings_cached(project_root: Path) -> pd.DataFrame:
    """Load bookings data with caching."""
    try:
        return load_hotel_bookings(project_root)
    except Exception as e:
        logger.error(f"Error loading bookings: {e}", exc_info=True)
        st.error(f"Error loading bookings: {e}")
        raise


# Load data
try:
    candidates, review_summary, cities = load_data_cached(root)
    bookings = load_bookings_cached(root)
    
    # Get unique values for dropdowns
    hotel_types = sorted(bookings["hotel"].dropna().unique().tolist()) if "hotel" in bookings.columns else []
    market_segments = sorted(bookings["market_segment"].dropna().unique().tolist()) if "market_segment" in bookings.columns else []
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# Sidebar navigation
st.sidebar.title("Features")
api_base_url = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
feature = st.sidebar.radio(
    "Select Feature",
    ["Hotel Recommendations", "Price Forecast", "Cancellation Risk", 
     "Overbooking Risk", "Price Fairness", "Best Booking Window"],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("AI-powered hotel analysis tools for smarter travel decisions.")

# Main content area
st.title(f"🏨 {feature}")

# Feature: Hotel Recommendations
if feature == "Hotel Recommendations":
    st.markdown("Get AI-powered hotel recommendations with explainable reasoning")
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", options=[""] + cities)
        budget = st.number_input("Budget", min_value=0.0, value=6000.0, step=100.0)
    with col2:
        min_rating = st.number_input("Min Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
        adults = st.number_input("Adults", min_value=0, value=2, step=1)
    
    limit = st.slider("Number of Results", min_value=1, max_value=30, value=10)
    
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Finding best hotels..."):
            payload = {
                "city": city or None,
                "budget": float(budget) if budget > 0 else None,
                "min_rating": float(min_rating) if min_rating > 0 else None,
                "adults": int(adults) if adults > 0 else None,
                "limit": int(limit),
            }

            try:
                resp = _post_json(api_base_url, "/recommend", payload)
            except Exception as e:
                st.error(str(e))
                st.stop()

            results = resp.get("results") if isinstance(resp, dict) else None
            if results:
                df = pd.DataFrame(results)
                show_cols = [
                    c
                    for c in [
                        "hotel",
                        "city",
                        "hotel_class",
                        "adr",
                        "adr_low",
                        "adr_high",
                        "sentiment_score",
                        "avg_rating",
                        "n_reviews",
                        "score",
                        "reason",
                        "url",
                    ]
                    if c in df.columns
                ]
                st.dataframe(df[show_cols] if show_cols else df, use_container_width=True, hide_index=True)
            else:
                st.info("No hotels found matching your criteria. Try adjusting your filters.")

# Feature: Price Forecast
elif feature == "Price Forecast":
    st.markdown("Predict future hotel prices and get booking timing advice")
    
    col1, col2 = st.columns(2)
    with col1:
        forecast_city = st.selectbox("City", options=[""] + cities)
        hotel_class = st.number_input("Hotel Class", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
    with col2:
        check_in_date = st.date_input(
            "Check-in Date",
            value=datetime.now().date() + timedelta(days=30),
            min_value=datetime.now().date()
        )
        horizon_days = st.slider("Forecast Horizon (days)", min_value=1, max_value=90, value=7)
    
    if st.button("Get Forecast", type="primary", use_container_width=True):
        if not forecast_city:
            st.warning("Please select a city")
        else:
            with st.spinner("Analyzing price trends..."):
                payload = {
                    "city": forecast_city,
                    "hotel_class": float(hotel_class) if hotel_class > 0 else None,
                    "check_in_date": check_in_date.isoformat(),
                    "horizon_days": int(horizon_days),
                }

                try:
                    resp = _post_json(api_base_url, "/forecast/signal", payload)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                sig = resp.get("result") if isinstance(resp, dict) else None
                if sig:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Trend", str(sig.get("trend", "")))
                    col2.metric("Expected Change", str(sig.get("expected_change", "")))
                    conf = sig.get("confidence")
                    col3.metric("Confidence", f"{float(conf):.1%}" if conf is not None else "")
                    col4.metric("Volatility", str(sig.get("hotel_price_volatility", "")))
                    
                    booking_advice = str(sig.get("booking_advice", ""))
                    advice_text = booking_advice.replace("_", " ").title()
                    if "BOOK_NOW" in booking_advice:
                        st.success(f"**Booking Advice**: {advice_text}")
                    elif "WAIT" in booking_advice:
                        st.warning(f"**Booking Advice**: {advice_text}")
                    else:
                        st.info(f"**Booking Advice**: {advice_text}")
                else:
                    st.error("Could not generate forecast. Please check your inputs.")

# Feature: Cancellation Risk
elif feature == "Cancellation Risk":
    st.markdown("Assess cancellation probability based on historical patterns")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_hotel_type = st.selectbox("Hotel Type", options=[""] + hotel_types)
    with col2:
        risk_market_segment = st.selectbox("Market Segment", options=[""] + market_segments)
    with col3:
        risk_lead_time = st.number_input("Lead Time (days)", min_value=0, value=30, step=1)
    
    if st.button("Analyze Risk", type="primary", use_container_width=True):
        if not risk_hotel_type or not risk_market_segment:
            st.warning("Please select hotel type and market segment")
        else:
            with st.spinner("Analyzing..."):
                payload = {
                    "hotel_type": risk_hotel_type,
                    "market_segment": risk_market_segment,
                    "lead_time": int(risk_lead_time),
                }

                try:
                    resp = _post_json(api_base_url, "/risk/cancellation", payload)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                risk = resp.get("result") if isinstance(resp, dict) else None
                if risk:
                    cancellation_rate = float(risk.get("cancellation_rate", 0.0))
                    cancellation_pct = cancellation_rate * 100
                    col1, col2 = st.columns(2)
                    col1.metric("Cancellation Rate", f"{cancellation_pct:.1f}%")
                    col2.metric("Sample Size", f"{risk.get('total_bookings', '')} bookings")
                    
                    advice = str(risk.get("advice", ""))
                    if cancellation_rate >= 0.30:
                        st.error(advice)
                    elif cancellation_rate >= 0.20:
                        st.warning(advice)
                    else:
                        st.info(advice)
                else:
                    st.error("Could not compute risk. Try different values.")

# Feature: Overbooking Risk
elif feature == "Overbooking Risk":
    st.markdown("Evaluate room reassignment and waitlist risks")
    
    col1, col2 = st.columns(2)
    with col1:
        over_hotel_type = st.selectbox("Hotel Type", options=[""] + hotel_types)
        over_month = st.selectbox("Arrival Month", options=[""] + months)
    with col2:
        over_market_segment = st.selectbox("Market Segment (Optional)", options=[""] + market_segments)
        is_repeated_guest = st.checkbox("Repeated Guest")
    
    prev_cancellations = st.number_input("Previous Cancellations", min_value=0, value=0)
    
    if st.button("Analyze Risk", type="primary", use_container_width=True):
        if not over_hotel_type or not over_month:
            st.warning("Please select hotel type and arrival month")
        else:
            with st.spinner("Analyzing..."):
                payload = {
                    "hotel_type": over_hotel_type,
                    "arrival_month": over_month,
                    "market_segment": over_market_segment if over_market_segment else None,
                    "is_repeated_guest": 1 if is_repeated_guest else 0,
                    "previous_cancellations": int(prev_cancellations),
                }

                try:
                    resp = _post_json(api_base_url, "/risk/overbooking", payload)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                over_risk = resp.get("result") if isinstance(resp, dict) else None
                if over_risk:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Level", str(over_risk.get("risk_level", "")).upper())
                    col2.metric("Reassignment Rate", f"{float(over_risk.get('reassignment_rate', 0.0)):.1%}")
                    col3.metric("Waitlist Rate", f"{float(over_risk.get('waiting_list_rate', 0.0)):.1%}")
                    
                    level = str(over_risk.get("risk_level", "")).lower()
                    advice = str(over_risk.get("advice", ""))
                    if level == "high":
                        st.error(advice)
                    elif level == "medium":
                        st.warning(advice)
                    else:
                        st.info(advice)
                else:
                    st.error("Could not compute risk. Try different values.")

# Feature: Price Fairness
elif feature == "Price Fairness":
    st.markdown("Compare current price against historical averages")
    
    col1, col2 = st.columns(2)
    with col1:
        fair_hotel_type = st.selectbox("Hotel Type", options=[""] + hotel_types)
        fair_month = st.selectbox("Arrival Month", options=[""] + months)
    with col2:
        fair_current_price = st.number_input("Current Price", min_value=0.0, value=100.0, step=10.0)
        fair_hotel_class = st.number_input("Hotel Class (Optional)", min_value=0.0, max_value=5.0, value=3.5, step=0.5)
    
    if st.button("Analyze Price", type="primary", use_container_width=True):
        if not fair_hotel_type or not fair_month:
            st.warning("Please select hotel type and arrival month")
        else:
            with st.spinner("Analyzing..."):
                payload = {
                    "hotel_type": fair_hotel_type,
                    "arrival_month": fair_month,
                    "current_price": float(fair_current_price),
                    "hotel_class": float(fair_hotel_class) if fair_hotel_class > 0 else None,
                }

                try:
                    resp = _post_json(api_base_url, "/advice/price_fairness", payload)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                fair_result = resp.get("result") if isinstance(resp, dict) else None
                if fair_result:
                    col1, col2 = st.columns(2)
                    col1.metric("Current Price", f"${float(fair_result.get('current_price', 0.0)):.2f}")
                    col2.metric("Expected Price", f"${float(fair_result.get('expected_price', 0.0)):.2f}")
                    
                    pct_diff = float(fair_result.get("pct_diff", 0.0))
                    st.metric("Difference", f"{abs(pct_diff):.1f}% {'below' if pct_diff < 0 else 'above'}")
                    
                    color = str(fair_result.get("color", ""))
                    label = str(fair_result.get("label", ""))
                    message = str(fair_result.get("message", ""))
                    if color == "green":
                        st.success(f"**{label}**: {message}")
                    elif color == "yellow":
                        st.warning(f"**{label}**: {message}")
                    else:
                        st.error(f"**{label}**: {message}")
                else:
                    st.error("Could not analyze price. Try different values.")

# Feature: Best Booking Window
elif feature == "Best Booking Window":
    st.markdown("Find optimal lead time for best prices")
    
    col1, col2 = st.columns(2)
    with col1:
        window_hotel_type = st.selectbox("Hotel Type", options=[""] + hotel_types)
    with col2:
        window_month = st.selectbox("Arrival Month", options=[""] + months)
    
    min_samples = st.slider("Minimum Samples", min_value=50, max_value=500, value=200, step=50)
    
    if st.button("Find Best Window", type="primary", use_container_width=True):
        if not window_hotel_type or not window_month:
            st.warning("Please select hotel type and arrival month")
        else:
            with st.spinner("Analyzing..."):
                payload = {
                    "hotel_type": window_hotel_type,
                    "arrival_month": window_month,
                    "min_samples": int(min_samples),
                }

                try:
                    resp = _post_json(api_base_url, "/advice/best_booking_window", payload)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

                window_result = resp.get("result") if isinstance(resp, dict) else None
                if window_result:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Recommended Window", str(window_result.get("recommended_window_days", "")))
                    col2.metric("Best Price", f"${float(window_result.get('min_median_adr', 0.0)):.2f}")
                    col3.metric("Confidence", f"{float(window_result.get('confidence', 0.0)):.1%}")
                    
                    st.info(str(window_result.get("message", "")))
                else:
                    st.error("Could not find booking window. Try lowering minimum samples.")
