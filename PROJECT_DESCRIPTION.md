# Project Description — AI-Powered Smart Travel & Stay Optimizer

## Overview
This project is an end-to-end **hotel decision intelligence system** designed to resemble real-world travel platforms (e.g., OYO, Booking.com, Expedia-style flows). It combines:

- **Explainable hotel recommendations**
- **Price forecasting (signal)** for booking-timing advice
- **Risk analytics** (cancellation + overbooking proxy)
- **Price fairness** and **best booking window** advisory insights

The codebase is split into a **FastAPI backend** (product-like service layer) and a **Streamlit frontend** (dashboard-style client). The analytics logic lives in `src/` as reusable modules.

## Architecture

### Components
- **Streamlit UI (`streamlit_app.py`)**
  - Provides a single dashboard-style interface.
  - Calls the backend via HTTP (configurable base URL).

- **FastAPI service (`api/app.py`)**
  - Loads data once at startup.
  - Exposes endpoints for all features.
  - Returns JSON responses intended for UI consumption.

- **Analytics modules (`src/`)**
  - Pure Python/Pandas analytics and heuristics.
  - Structured outputs using dataclasses in several modules.

### Data Flow (high level)
1. FastAPI starts and loads datasets via `src/data_loader.py`.
2. FastAPI precomputes reusable tables (risk/insight tables) from bookings data.
3. Streamlit collects user inputs and calls FastAPI endpoints.
4. UI renders results: metrics, messages, and tabular output.

## Data Sources

### 1) `hotel_bookings.csv` (root)
Used for:
- Cancellation risk analytics
- Overbooking proxy analytics
- Price fairness baseline (expected ADR)
- Best booking window analytics
- Price forecasting baseline time series

Key columns used (varies by feature):
- `hotel`
- `market_segment`
- `lead_time`
- `is_canceled`
- `arrival_date_year`, `arrival_date_month`, `arrival_date_day_of_month`
- `adr`
- `reserved_room_type`, `assigned_room_type`
- `days_in_waiting_list`

### 2) `data/offerings.csv`
Used for:
- Recommendation candidate hotels

Typical columns:
- `id` (hotel/offering identifier)
- `name`
- `hotel_class`
- `address` (contains locality/city-like values)

### 3) Reviews data (project uses review summaries)
Used for:
- Recommendation quality signals (sentiment / ratings / pros / cons)

(Exact file may vary depending on preprocessing pipeline; the backend uses `load_review_summaries(...)` / `load_all_data(...)`.)

## Product Features (What they do and how they are computed)

### Feature 1 — Hotel Recommendations (Explainable)

**UI**: “Hotel Recommendations”

**API**: `POST /recommend`

**Goal**: Return a ranked list of hotels with a human-readable explanation (“reason”) for each result.

**Inputs (typical)**
- `city` (optional)
- `budget` (optional)
- `min_rating` (optional)
- `adults` (optional)
- `limit` (default 10)

**How it is computed (high level)**
- Candidate hotels are prepared from `offerings.csv` and enriched with:
  - **Estimated ADR** and/or **price band** (`adr_low`, `adr_high`)
  - Review summary features (if available):
    - `sentiment_score`
    - `avg_rating`
    - `n_reviews`
    - `pros`, `cons`
- Ranking is computed by `src/recommender.py` (e.g., `recommend_explainable(...)`).
- The “reason” field is generated from available signals, typically combining:
  - sentiment/ratings quality
  - budget fit
  - general relevance

**Important assumptions / notes**
- Some signals are estimates because not all datasets contain all real-world pricing fields.
- The recommendation output is **explainable** by design: the UI always shows supporting fields.

---

### Feature 2 — Price Forecast (Signal)

**UI**: “Price Forecast”

**API**: `POST /forecast/signal`

**Goal**: Provide a compact booking-timing signal instead of a full forecast table.

**Outputs**
- `trend` (e.g., increasing/decreasing/stable)
- `expected_change` (percentage string)
- `confidence` (0–1)
- `hotel_price_volatility` (bucket/label)
- `booking_advice` (e.g., BOOK_NOW / WAIT / WATCH)

**Inputs**
- `city`
- `hotel_class` (optional)
- `check_in_date` (ISO date string)
- `horizon_days` (default 7)

**How it is computed** (`src/forecast.py`)
- A historical ADR time series is built from bookings:
  - ADR cleaned/coerced to numeric
  - aggregated by date (median/robust aggregation)
- A trend model is fit (lightweight linear trend over time).
- Forecast for the next `horizon_days` is generated.
- City and hotel-class multipliers are applied so the signal is **context-aware**.
- `expected_change` is derived from the scaled forecast start vs end.
- Confidence is derived heuristically from:
  - sample volume and stability
  - volatility / dispersion

**Why this is useful**
- It mimics how a real travel product would communicate price direction: a user-friendly signal.

---

### Feature 3 — Cancellation Risk

**UI**: “Cancellation Risk”

**API**: `POST /risk/cancellation`

**Goal**: Estimate cancellation probability for a booking context.

**Inputs**
- `hotel_type` (e.g., City Hotel / Resort Hotel)
- `market_segment`
- `lead_time` (days before arrival)

**How it is computed** (`src/cancellation_risk.py`)
1. A **risk table** is precomputed using historical bookings:
   - grouped by hotel type + market segment + lead-time bucket
   - cancellation rate = `mean(is_canceled)`
   - sample sizes recorded for reliability
2. At request time, the system:
   - finds the closest relevant group / bucket
   - returns:
     - `cancellation_rate`
     - `canceled_bookings`, `total_bookings`
     - a short `advice` string

**Assumptions / limitations**
- This is a cohort-based estimator (not a per-user ML classifier).
- Useful for product transparency and risk-aware UX.

---

### Feature 4 — Overbooking Risk (Proxy)

**UI**: “Overbooking Risk”

**API**: `POST /risk/overbooking`

**Goal**: Provide a practical proxy for operational disruption risk (room reassignment / waitlist pressure).

**Inputs**
- `hotel_type`
- `arrival_month`
- `market_segment` (optional)
- `is_repeated_guest` (optional)
- `previous_cancellations` (optional)

**How it is computed** (`src/overbooking_risk.py`)
1. Precompute a proxy table from bookings:
   - `reassigned = reserved_room_type != assigned_room_type`
   - `waitlisted = days_in_waiting_list > 0`
   - aggregate by `hotel`, `arrival_date_month`, `market_segment`
2. Compute proxy score:
   - `risk_score = 0.7 * reassignment_rate + 0.3 * waiting_list_rate`
3. Map score to levels:
   - `high` if `>= 0.25`
   - `medium` if `>= 0.12`
   - else `low`
4. Apply small advisory adjustments:
   - repeated guest: slightly lower risk
   - many previous cancellations: slightly higher risk

**Important resume-friendly note**
- This is explicitly a **proxy** (real overbooking requires inventory/capacity systems).
- Still valuable because it demonstrates operational thinking and pragmatic risk modeling.

---

### Feature 5 — Price Fairness

**UI**: “Price Fairness”

**API**: `POST /advice/price_fairness`

**Goal**: Answer “Is this price good for similar hotels in this season?”

**Inputs**
- `hotel_type`
- `arrival_month`
- `current_price`
- `hotel_class` (optional)

**How it is computed** (`src/price_insights.py`)
1. Precompute expected ADR baseline:
   - group by `hotel` + `arrival_date_month`
   - `expected_adr = median(adr)`
2. At request time:
   - select the best matching baseline row
   - optionally adjust baseline by hotel class:
     - `class_mult = clip(hotel_class / class_base, 0.6, 2.0)`
     - `expected_adj = expected_adr * class_mult`
   - compute fairness ratio:
     - `ratio = current_price / expected_adj`
     - `pct_diff = (ratio - 1) * 100`
3. Map ratio to labels/colors:
   - `<= 0.92` → “Great deal” (green)
   - `<= 1.05` → “Fair price” (green)
   - `<= 1.15` → “Slightly high” (yellow)
   - else → “Overpriced” (red)

---

### Feature 6 — Best Booking Window

**UI**: “Best Booking Window”

**API**: `POST /advice/best_booking_window`

**Goal**: Recommend how far in advance to book for lowest typical prices.

**Inputs**
- `hotel_type`
- `arrival_month`
- `min_samples` (minimum data per bucket; default 200)

**How it is computed** (`src/price_insights.py`)
1. Bucket lead time into ranges:
   - `0-7`, `8-14`, `15-21`, `22-28`, ..., `181+`
2. Precompute booking-window table:
   - group by `hotel`, `arrival_date_month`, `lead_time_bucket`
   - `median_adr` per bucket and sample size `n`
3. At request time:
   - filter buckets with `n >= min_samples`
   - pick bucket with **minimum median ADR**
4. Confidence is a heuristic combining:
   - sample size strength
   - separation from overall median bucket ADR

## API Summary

### Core endpoints
- `GET /health`
- `GET /meta/cities`
- `GET /meta/stats`

### Product features
- `POST /recommend`
- `POST /forecast/signal`
- `POST /risk/cancellation`
- `POST /risk/overbooking`
- `POST /advice/price_fairness`
- `POST /advice/best_booking_window`

## Running the Project (local)

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Start the API
```bash
uvicorn api.app:app --reload
```

### 3) Start Streamlit
```bash
streamlit run streamlit_app.py
```

### 4) Use the UI
- Open Streamlit in your browser.
- Ensure the sidebar `API Base URL` is set to your running FastAPI server.

## Limitations and Responsible Use
- Several features are heuristic/proxy-based due to limited real-world inventory/pricing data.
- Outputs are intended as **decision support**, not guaranteed outcomes.
- This is a resume-grade portfolio project: correctness, transparency, and product thinking are prioritized.
