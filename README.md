# TravelIQ

AI-powered hotel analysis tools for smarter travel decisions — recommendations, price positioning, cancellation risk, and booking timing in one place.

## Features

| Feature | Description |
|---|---|
| **Booking Advisor** | One-shot trip analysis: recommendations + price seasonality + risks + booking timing |
| **Hotel Recommendations** | Ranked hotels scored on price, sentiment, rating, and review volume |
| **Cancellation Risk** | Historical cancellation probability by hotel type, market segment, and lead time |
| **Price Fairness** | Compare a quoted price against seasonal historical averages |
| **Best Booking Window** | How many days in advance to book for the lowest price |
| **Seasonal Price Heatmap** | Calendar-style view of cheap vs expensive months by hotel type |

## Project Structure

```
TravelIQ/
├── api/
│   └── app.py                    # FastAPI backend
├── data/
│   ├── Hotel Reviews.csv         # Raw hotel reviews (10,000 rows, 1,853 hotels)
│   ├── offerings.csv             # Hotel metadata (id, name, city, class, url)
│   └── hotel_review_summaries.csv  # Processed summaries with sentiment scores
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constants and environment variable overrides
│   ├── data_loader.py            # Centralised data loading
│   ├── utils.py                  # Shared utilities
│   ├── logger.py                 # Structured logging
│   ├── recommender.py            # Weighted scoring recommendation engine
│   ├── cancellation_risk.py      # Cohort-based cancellation risk tables
│   └── price_insights.py        # ADR tables, price fairness, booking windows
├── scripts/
│   └── build_data.py             # Pre-processes Hotel Reviews.csv → offerings.csv + summaries
├── tests/
│   └── test_analytics.py         # 37 unit tests for analytics modules
├── hotel_bookings.csv            # Historical booking data (119K rows) — project root
├── streamlit_app.py              # Streamlit frontend (no backend required for most features)
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Data

Two large raw data files are excluded from this repo (see `.gitignore`) and must be downloaded separately:

| File | Location | Source |
|---|---|---|
| `hotel_bookings.csv` | project root | [Kaggle — Hotel Booking Demand](https://www.kaggle.com/jessemostipak/hotel-booking-demand) (16 MB) |
| `data/Hotel Reviews.csv` | `data/` | [Kaggle — TripAdvisor Hotel Reviews](https://www.kaggle.com/andrewmvd/trip-advisor-hotel-reviews) (48 MB) |

The processed outputs (`data/offerings.csv`, `data/hotel_review_summaries.csv`) are included in the repo and ready to use. If you want to regenerate them from the raw reviews:

| File | Location | Description |
|---|---|---|
| `data/offerings.csv` | `data/` | Hotel metadata (id, name, city, hotel_class) — included |
| `data/hotel_review_summaries.csv` | `data/` | Pre-processed review summaries — included |

To (re-)build `offerings.csv` and `hotel_review_summaries.csv` from the raw reviews:

```bash
python scripts/build_data.py
```

### 3. Run the Streamlit App

```bash
python -m streamlit run streamlit_app.py
```

Access at: http://localhost:8501

The Streamlit app connects to the FastAPI backend for recommendation and risk queries. Start the backend first:

```bash
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

## API Reference

Base URL: `http://127.0.0.1:8000`

Interactive docs: http://127.0.0.1:8000/docs

### Endpoints

**`GET /health`** — Health check

**`GET /meta/cities`** — List of cities with available hotel data

**`GET /meta/stats`** — Dataset statistics

**`POST /recommend`** — Hotel recommendations
```json
{
  "city": "New York",
  "budget": 200.0,
  "min_rating": 4.0,
  "adults": 2,
  "hotel_type": "City Hotel",
  "limit": 10
}
```

**`POST /advisor/summary`** — All-in-one booking advisor (runs all analyses in one call)
```json
{
  "city": "New York",
  "hotel_type": "City Hotel",
  "arrival_month": "June",
  "check_in_date": "2026-06-15",
  "budget": 200.0,
  "limit": 5
}
```

**`POST /risk/cancellation`** — Cancellation risk by segment and lead time
```json
{
  "hotel_type": "Resort Hotel",
  "market_segment": "Online TA",
  "lead_time": 30
}
```

**`POST /advice/price_fairness`** — Price vs seasonal historical average
```json
{
  "hotel_type": "City Hotel",
  "arrival_month": "July",
  "current_price": 180.0,
  "hotel_class": 4.0
}
```

**`POST /advice/best_booking_window`** — Optimal days-in-advance to book
```json
{
  "hotel_type": "City Hotel",
  "arrival_month": "August",
  "min_samples": 200
}
```

## Configuration

Environment variable overrides (all optional):

| Variable | Default | Description |
|---|---|---|
| `TRAVEL_API_HOST` | `127.0.0.1` | API bind host |
| `TRAVEL_API_PORT` | `8000` | API bind port |

Key constants in `src/config.py`:
- `DEFAULT_CLASS_BASE = 3.5` — baseline hotel class for price normalisation

## How Recommendations Work

Hotels are scored using a weighted combination of four signals:

| Signal | Weight | Source |
|---|---|---|
| Price fit | 30% | How close the hotel ADR is to the user's budget |
| Sentiment | 30% | Review sentiment score (derived from ratings) |
| Rating | 25% | Average guest rating |
| Review volume | 15% | Log-normalised number of reviews |

Results include a human-readable reason string explaining why each hotel ranked where it did.

## Data Notes

- **Booking data** covers 2015–2017. Price seasonality, cancellation rates, and booking windows all reflect this period.
- **Price Fairness** comparisons are seasonal averages from this historical dataset — not live market rates.
- **Hotel Reviews** are from TripAdvisor; sentiment is derived from star ratings (rating / 5).

## Dependencies

Key packages:
- `fastapi`, `uvicorn` — REST API
- `streamlit` — interactive frontend
- `pandas`, `numpy` — data processing
- `scikit-learn` — price model (price_model.py)
- `plotly` — interactive charts (heatmap, lead time chart)
- `pydantic` — request validation

See `requirements.txt` for full pinned versions.
