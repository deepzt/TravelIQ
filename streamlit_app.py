from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.data_loader import load_all_data, load_hotel_bookings
from src.utils import get_project_root
from src.logger import logger


st.set_page_config(page_title="TravelIQ", layout="wide", initial_sidebar_state="expanded")

root = get_project_root()


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_json(base_url: str, path: str, payload: dict) -> dict:
    url = base_url.rstrip("/") + path
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json() if resp.text else {}
    except requests.HTTPError as e:
        raise RuntimeError(f"HTTP {e.response.status_code} from {url}: {e.response.text[:200]}")
    except requests.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to API at {url}. "
            "Make sure the backend is running: `python run_api.py`"
        )
    except requests.Timeout:
        raise RuntimeError(f"Request to {url} timed out after 30s.")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data_cached(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    try:
        candidates, review_summary, _, cities = load_all_data(project_root)
        return candidates, review_summary, cities
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        raise


@st.cache_data(show_spinner=False)
def load_bookings_cached(project_root: Path) -> pd.DataFrame:
    try:
        return load_hotel_bookings(project_root)
    except Exception as e:
        logger.error(f"Error loading bookings: {e}", exc_info=True)
        raise


@st.cache_data(show_spinner=False)
def load_booking_window_table(project_root: Path):
    from src.price_insights import compute_booking_window_table
    bk = load_hotel_bookings(project_root)
    return compute_booking_window_table(bk)


@st.cache_data(show_spinner=False)
def load_expected_adr_table(project_root: Path):
    from src.price_insights import compute_expected_adr_table
    bk = load_hotel_bookings(project_root)
    return compute_expected_adr_table(bk)


try:
    candidates, review_summary, cities = load_data_cached(root)
    bookings = load_bookings_cached(root)
    hotel_types = sorted(bookings["hotel"].dropna().unique().tolist()) if "hotel" in bookings.columns else []
    market_segments = sorted(bookings["market_segment"].dropna().unique().tolist()) if "market_segment" in bookings.columns else []
    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("TravelIQ")
api_base_url = st.sidebar.text_input("API Base URL", value="http://127.0.0.1:8000")
feature = st.sidebar.radio(
    "Navigate",
    options=[
        "Booking Advisor",
        "Hotel Recommendations",
        "Cancellation Risk",
        "Price Fairness",
        "Best Booking Window",
        "Seasonal Price Heatmap",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("AI-powered hotel analysis tools for smarter travel decisions.")

st.title(f"TravelIQ — {feature}")


# ---------------------------------------------------------------------------
# Market segment descriptions (Cancellation Risk tooltip)
# ---------------------------------------------------------------------------

_SEGMENT_INFO: dict[str, tuple[str, str]] = {
    "Online TA": (
        "Online Travel Agency",
        "Booked via platforms like Booking.com, Expedia, or Hotels.com. "
        "Typically has the highest cancellation rates — these platforms make it easy to book multiple options and cancel freely.",
    ),
    "Offline TA/TO": (
        "Offline Travel Agent / Tour Operator",
        "Booked through a traditional travel agent or tour operator (phone/in-person), often as part of a package deal. "
        "More committed than online bookings; moderate cancellation risk.",
    ),
    "Groups": (
        "Group Bookings",
        "Block reservations for events like weddings, conferences, or sports teams. "
        "Risk varies — large groups can cancel en masse, but often have stricter contracts.",
    ),
    "Direct": (
        "Direct Booking",
        "Booked directly with the hotel via their website, phone, or walk-in. "
        "Generally lower cancellation risk due to greater commitment and often non-refundable rates.",
    ),
    "Corporate": (
        "Corporate / Business Travel",
        "Negotiated rates through a company contract for business travellers. "
        "Low cancellation risk — bookings are tied to business schedules.",
    ),
    "Complementary": (
        "Complimentary Stay",
        "Free or comped stays — staff, VIP guests, press, or compensation bookings. "
        "Cancellation patterns are atypical; sample sizes are small.",
    ),
    "Aviation": (
        "Airline Crew",
        "Layover stays booked by airlines for crew. "
        "Very predictable — short stays on fixed schedules, low cancellation risk.",
    ),
    "Undefined": (
        "Undefined",
        "Segment not recorded. Only 2 rows in the dataset — effectively ignorable.",
    ),
}

_SEGMENT_HELP = "\n\n".join(
    f"**{k}** ({v[0]}): {v[1]}" for k, v in _SEGMENT_INFO.items()
)


# ---------------------------------------------------------------------------
# Recommendation table formatting
# ---------------------------------------------------------------------------

_COL_LABELS = {
    "hotel": "Hotel",
    "city": "City",
    "hotel_class": "Stars",
    "adr": "Est. Price/Night",
    "adr_low": "Price Low",
    "adr_high": "Price High",
    "price_confidence": "Price Confidence",
    "sentiment_score": "Sentiment",
    "avg_rating": "Rating",
    "n_reviews": "Reviews",
    "score": "Match Score",
    "reason": "Why Recommended",
    "url": "Link",
}


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _format_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    show_cols = [c for c in _COL_LABELS if c in df.columns]
    out = df[show_cols].copy().rename(columns=_COL_LABELS)

    for col, fmt in [
        ("Est. Price/Night", "${:.0f}"),
        ("Price Low", "${:.0f}"),
        ("Price High", "${:.0f}"),
        ("Stars", "{:.1f}"),
        ("Rating", "{:.2f}"),
        ("Match Score", "{:.3f}"),
        ("Sentiment", "{:.0%}"),
    ]:
        if col in out.columns:
            out[col] = out[col].apply(lambda v, f=fmt: f.format(float(v)) if _is_numeric(v) else v)

    if "Reviews" in out.columns:
        out["Reviews"] = out["Reviews"].apply(lambda v: f"{int(float(v)):,}" if _is_numeric(v) else v)

    return out


# ---------------------------------------------------------------------------
# Feature: Booking Advisor
# ---------------------------------------------------------------------------

if feature == "Booking Advisor":
    st.markdown("Get a complete trip analysis in one shot — recommendations, price seasonality, risks, and booking timing.")

    col1, col2 = st.columns(2)
    with col1:
        adv_city = st.selectbox("City", options=[""] + cities)
        adv_budget = st.number_input("Budget per night ($)", min_value=0.0, value=200.0, step=50.0)
        adv_hotel_type = st.selectbox("Hotel type", options=[""] + hotel_types)
    with col2:
        adv_check_in = st.date_input(
            "Check-in date",
            value=datetime.now().date() + timedelta(days=30),
            min_value=datetime.now().date(),
        )
        adv_month = st.selectbox("Arrival month", options=months, index=adv_check_in.month - 1)
        adv_min_rating = st.number_input("Minimum rating (0 = any)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)

    if st.button("Run Full Analysis", type="primary", width="stretch"):
        if not adv_hotel_type or not adv_month:
            st.warning("Please select at least a hotel type and arrival month.")
        else:
            with st.spinner("Running all analyses..."):
                payload = {
                    "city": adv_city or None,
                    "hotel_type": adv_hotel_type,
                    "arrival_month": adv_month,
                    "check_in_date": adv_check_in.isoformat(),
                    "budget": float(adv_budget) if adv_budget > 0 else None,
                    "min_rating": float(adv_min_rating) if adv_min_rating > 0 else None,
                    "limit": 5,
                }
                try:
                    resp = _post_json(api_base_url, "/advisor/summary", payload)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

            cancel = resp.get("cancellation_risk") or {}
            fairness = resp.get("price_fairness") or {}
            window = resp.get("booking_window") or {}
            cancel_rate = float(cancel.get("cancellation_rate", 0.0))

            # Verdict banner — based on cancellation risk and price fairness only
            if cancel_rate < 0.20 and fairness.get("color") in ("green", None):
                st.success("**Verdict: Good conditions.** Low cancellation risk and price looks reasonable.")
            elif cancel_rate >= 0.35:
                st.warning("**Verdict: Book carefully.** High cancellation risk for this segment — review policy before confirming.")
            else:
                st.info("**Verdict: Conditions are stable.** Check the details below before booking.")

            st.markdown("---")

            # Summary metrics
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Seasonal Price Position", str(fairness.get("label", "—")),
                       delta=f"{fairness.get('pct_diff', 0):+.1f}% vs avg" if fairness.get("pct_diff") is not None else None,
                       delta_color="inverse")
            mc2.metric("Cancellation Risk", f"{cancel_rate:.0%}" if cancel else "—")
            mc3.metric("Best Lead Time", f"{window.get('recommended_window_days', '—')} days" if window else "—")

            st.markdown("---")

            if window:
                st.markdown(f"**Best time to book:** {window.get('message', '')}")

            recs = resp.get("recommendations", [])
            if recs:
                st.markdown("### Top Hotels")
                rec_df = pd.DataFrame(recs)
                st.dataframe(_format_recommendations(rec_df), width="stretch", hide_index=True)
            else:
                st.info("No hotel recommendations available for this city/filters.")

            with st.expander("Cancellation risk details"):
                if cancel:
                    st.markdown(f"Rate: **{cancel_rate:.0%}** ({cancel.get('total_bookings', 0):,} bookings sampled)")
                    st.markdown(cancel.get("advice", ""))
                else:
                    st.markdown("No data.")


# ---------------------------------------------------------------------------
# Feature: Hotel Recommendations
# ---------------------------------------------------------------------------

elif feature == "Hotel Recommendations":
    st.markdown("Get AI-powered hotel recommendations with explainable reasoning.")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", options=[""] + cities)
        budget = st.number_input("Max budget per night ($)", min_value=0.0, value=200.0, step=50.0)
    with col2:
        min_rating = st.number_input("Minimum rating (0 = any)", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
        adults = st.number_input("Adults", min_value=1, value=2, step=1)

    limit = st.slider("Number of results", min_value=1, max_value=30, value=10)

    if st.button("Get Recommendations", type="primary", width="stretch"):
        with st.spinner("Finding best hotels..."):
            payload = {
                "city": city.strip() or None,
                "budget": float(budget) if budget > 0 else None,
                "min_rating": float(min_rating) if min_rating > 0 else None,
                "adults": int(adults),
                "limit": int(limit),
            }
            try:
                resp = _post_json(api_base_url, "/recommend", payload)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()

            results = resp.get("results") if isinstance(resp, dict) else None
            if results:
                df = pd.DataFrame(results)
                st.success(f"Found {len(results)} hotels matching your criteria.")
                st.dataframe(_format_recommendations(df), width="stretch", hide_index=True)

                if "pros" in df.columns and len(df) > 0:
                    top = df.iloc[0]
                    with st.expander(f"Top pick details: {top.get('hotel', '')}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Pros**")
                            pros = top.get("pros", [])
                            if isinstance(pros, str):
                                import ast
                                try:
                                    pros = ast.literal_eval(pros)
                                except Exception:
                                    pros = [pros]
                            for p in (pros or []):
                                st.markdown(f"- {p}")
                        with c2:
                            st.markdown("**Cons**")
                            cons = top.get("cons", [])
                            if isinstance(cons, str):
                                import ast
                                try:
                                    cons = ast.literal_eval(cons)
                                except Exception:
                                    cons = [cons]
                            for c in (cons or []):
                                st.markdown(f"- {c}")
            else:
                st.warning("No hotels found matching your criteria. Try adjusting your filters.")


# ---------------------------------------------------------------------------
# Feature: Cancellation Risk
# ---------------------------------------------------------------------------

elif feature == "Cancellation Risk":
    st.markdown("Estimate cancellation probability based on historical booking patterns.")

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_hotel_type = st.selectbox("Hotel type", options=[""] + hotel_types)
    with col2:
        risk_market_segment = st.selectbox(
            "Market segment",
            options=[""] + market_segments,
            help=_SEGMENT_HELP,
        )
        if risk_market_segment and risk_market_segment in _SEGMENT_INFO:
            label, desc = _SEGMENT_INFO[risk_market_segment]
            st.caption(f"**{label}** — {desc}")
    with col3:
        risk_lead_time = st.number_input("Lead time (days before arrival)", min_value=0, value=30, step=1)

    if st.button("Analyse Risk", type="primary", width="stretch"):
        if not risk_hotel_type or not risk_market_segment:
            st.warning("Please select both hotel type and market segment.")
        else:
            with st.spinner("Analysing..."):
                payload = {
                    "hotel_type": risk_hotel_type,
                    "market_segment": risk_market_segment,
                    "lead_time": int(risk_lead_time),
                }
                try:
                    resp = _post_json(api_base_url, "/risk/cancellation", payload)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

                risk = resp.get("result") if isinstance(resp, dict) else None
                if risk:
                    cancellation_rate = float(risk.get("cancellation_rate", 0.0))
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Cancellation Rate", f"{cancellation_rate:.0%}")
                    col2.metric("Sample Size", f"{risk.get('total_bookings', '—'):,}" if _is_numeric(risk.get('total_bookings')) else "—")
                    col3.metric("Lead Time Bucket", str(risk.get("lead_time_bucket", "—")))

                    advice = str(risk.get("advice", ""))
                    if cancellation_rate >= 0.30:
                        st.error(advice)
                    elif cancellation_rate >= 0.20:
                        st.warning(advice)
                    else:
                        st.success(advice)
                else:
                    st.warning("No matching data found. Try different inputs.")


# ---------------------------------------------------------------------------
# Feature: Price Fairness
# ---------------------------------------------------------------------------

elif feature == "Price Fairness":
    st.markdown("See how your price compares to the seasonal average for this hotel type.")
    st.caption(
        "Seasonal averages are derived from historical booking data (2015-2017). "
        "Use this to understand **relative** price positioning within a season, "
        "not as a direct comparison to today's market rates."
    )

    col1, col2 = st.columns(2)
    with col1:
        fair_hotel_type = st.selectbox("Hotel type", options=[""] + hotel_types)
        fair_month = st.selectbox("Arrival month", options=[""] + months)
    with col2:
        fair_current_price = st.number_input("Price you're seeing ($/night)", min_value=1.0, value=150.0, step=10.0)
        fair_hotel_class = st.number_input("Hotel class / stars (optional, 0 = unknown)", min_value=0.0, max_value=5.0, value=3.5, step=0.5)

    if st.button("Analyse Price", type="primary", width="stretch"):
        if not fair_hotel_type or not fair_month:
            st.warning("Please select hotel type and arrival month.")
        elif fair_current_price <= 0:
            st.warning("Please enter a valid price.")
        else:
            with st.spinner("Analysing..."):
                payload = {
                    "hotel_type": fair_hotel_type,
                    "arrival_month": fair_month,
                    "current_price": float(fair_current_price),
                    "hotel_class": float(fair_hotel_class) if fair_hotel_class > 0 else None,
                }
                try:
                    resp = _post_json(api_base_url, "/advice/price_fairness", payload)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

                fair_result = resp.get("result") if isinstance(resp, dict) else None
                if fair_result:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Your Price", f"${float(fair_result.get('current_price', 0)):.2f}")
                    col2.metric("Historical Seasonal Avg", f"${float(fair_result.get('expected_price', 0)):.2f}")
                    pct_diff = float(fair_result.get("pct_diff", 0.0))
                    col3.metric("vs Seasonal Avg", f"{abs(pct_diff):.1f}% {'above' if pct_diff > 0 else 'below'}")

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
                    st.warning("No matching data found. Try different inputs.")


# ---------------------------------------------------------------------------
# Feature: Best Booking Window
# ---------------------------------------------------------------------------

elif feature == "Best Booking Window":
    st.markdown("Find the optimal number of days in advance to book for the lowest price.")

    col1, col2 = st.columns(2)
    with col1:
        window_hotel_type = st.selectbox("Hotel type", options=[""] + hotel_types)
    with col2:
        window_month = st.selectbox("Arrival month", options=[""] + months)

    min_samples = st.slider(
        "Minimum data points per bucket",
        min_value=50, max_value=500, value=200, step=50,
        help="Higher values give more reliable results but may return nothing for rare combinations.",
    )

    if st.button("Find Best Window", type="primary", width="stretch"):
        if not window_hotel_type or not window_month:
            st.warning("Please select hotel type and arrival month.")
        else:
            with st.spinner("Analysing..."):
                payload = {
                    "hotel_type": window_hotel_type,
                    "arrival_month": window_month,
                    "min_samples": int(min_samples),
                }
                try:
                    resp = _post_json(api_base_url, "/advice/best_booking_window", payload)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()

                window_result = resp.get("result") if isinstance(resp, dict) else None
                if window_result:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Booking Window", f"{window_result.get('recommended_window_days', '—')} days ahead")
                    col2.metric("Typical Price", f"${float(window_result.get('min_median_adr', 0)):.0f}/night")
                    col3.metric("Confidence", f"{float(window_result.get('confidence', 0)):.0%}")
                    st.info(str(window_result.get("message", "")))
                else:
                    st.warning(
                        "No result found. Try lowering the minimum samples threshold, "
                        "or check that you have data for this hotel type and month combination."
                    )

    # Lead Time vs Price Chart — always shown once hotel type and month are selected
    if window_hotel_type and window_month:
        st.markdown("---")
        st.markdown("### Price by Booking Window")
        st.caption("How median nightly price changes depending on how far in advance you book.")

        try:
            wt = load_booking_window_table(root)
            mask = (
                wt["hotel_type"].str.lower() == window_hotel_type.lower()
            ) & (
                wt["arrival_date_month"].str.lower() == window_month.lower()
            )
            chart_df = wt[mask].copy()

            if not chart_df.empty:
                _bucket_order = ["0-7", "8-14", "15-21", "22-28", "29-60",
                                 "61-90", "91-120", "121-180", "181+"]
                chart_df["_order"] = chart_df["lead_time_bucket"].apply(
                    lambda b: _bucket_order.index(b) if b in _bucket_order else 99
                )
                chart_df = chart_df.sort_values("_order")
                best_bucket = chart_df.loc[chart_df["median_adr"].idxmin(), "lead_time_bucket"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=chart_df["lead_time_bucket"],
                    y=chart_df["median_adr"].round(2),
                    mode="lines+markers",
                    line=dict(color="#4C78A8", width=2),
                    marker=dict(
                        size=[14 if b == best_bucket else 7 for b in chart_df["lead_time_bucket"]],
                        color=["#E45756" if b == best_bucket else "#4C78A8" for b in chart_df["lead_time_bucket"]],
                    ),
                    hovertemplate="<b>%{x} days ahead</b><br>Median price: $%{y:.0f}/night<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="Days Booked in Advance",
                    yaxis_title="Median Price ($/night)",
                    height=350,
                    margin=dict(t=20, b=40),
                    plot_bgcolor="white",
                    yaxis=dict(gridcolor="#f0f0f0"),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    f"Red dot = cheapest window ({best_bucket} days ahead). "
                    f"Based on {int(chart_df['n'].sum()):,} historical bookings."
                )
            else:
                st.info("Not enough data to plot chart for this combination.")
        except Exception as e:
            st.info(f"Chart unavailable: {e}")


# ---------------------------------------------------------------------------
# Feature: Seasonal Price Heatmap
# ---------------------------------------------------------------------------

elif feature == "Seasonal Price Heatmap":
    st.markdown("See which months are cheapest or most expensive for each hotel type.")

    try:
        adr_table = load_expected_adr_table(root)
        _month_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]

        pivot = adr_table.pivot_table(
            index="hotel_type",
            columns="arrival_date_month",
            values="expected_adr",
            aggfunc="median",
        )
        pivot = pivot[[m for m in _month_order if m in pivot.columns]]

        fig = px.imshow(
            pivot,
            labels=dict(x="Month", y="Hotel Type", color="Median ADR ($/night)"),
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            text_auto=".0f",
        )
        fig.update_layout(
            height=300,
            margin=dict(t=30, b=40),
            coloraxis_colorbar=dict(title="$/night"),
            xaxis_title="",
            yaxis_title="",
        )
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, width="stretch")
        st.caption("Green = cheaper months, Red = more expensive. Values are median nightly ADR from 119K historical bookings.")

        st.markdown("### Best month to travel by hotel type")
        for ht in pivot.index:
            cheapest = pivot.loc[ht].idxmin()
            cheapest_val = pivot.loc[ht].min()
            priciest = pivot.loc[ht].idxmax()
            priciest_val = pivot.loc[ht].max()
            st.markdown(
                f"**{ht}** — Cheapest: {cheapest} (~${cheapest_val:.0f}/night) &nbsp;|&nbsp; "
                f"Most expensive: {priciest} (~${priciest_val:.0f}/night)"
            )
    except Exception as e:
        st.error(f"Could not build heatmap: {e}")
