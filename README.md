# AI-Powered Smart Travel & Stay Optimizer

An intelligent hotel recommendation system with price forecasting, cancellation risk analysis, and explainable recommendations.

## Features

- **Hotel Recommendations**: AI-powered recommendations based on budget, ratings, sentiment, and location
- **Price Forecasting**: Predict future hotel prices to optimize booking timing
- **Cancellation Risk Analysis**: Assess cancellation probability based on booking patterns
- **Overbooking Risk**: Evaluate room reassignment and waitlist risks
- **Price Fairness**: Compare current prices against historical averages
- **Explainable AI**: Transparent reasoning for each recommendation

## Project Structure

```
Travel/
├── api/                 # FastAPI backend
│   └── app.py          # API endpoints
├── data/               # Data files (CSV)
│   ├── hotel_bookings.csv
│   ├── offerings.csv
│   └── hotel_review_summaries.csv
├── notebooks/          # Jupyter notebooks for analysis
│   ├── 01_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_recommendation.ipynb
│   ├── 04_price_prediction.ipynb
│   └── 05_review_nlp.ipynb
├── src/                # Source code
│   ├── __init__.py
│   ├── config.py       # Configuration and constants
│   ├── data_loader.py   # Centralized data loading
│   ├── utils.py         # Utility functions
│   ├── logger.py        # Logging configuration
│   ├── recommender.py   # Recommendation engine
│   ├── forecast.py      # Price forecasting
│   ├── cancellation_risk.py
│   ├── overbooking_risk.py
│   └── price_insights.py
├── web/                # Frontend
│   └── index.html      # React-based UI
├── streamlit_app.py    # Streamlit demo app
├── requirements.txt    # Python dependencies
└── README.md
```

## Quickstart

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Ensure your data files are in the `data/` directory:
- `hotel_bookings.csv` - Historical booking data
- `offerings.csv` - Hotel offerings and metadata
- `hotel_review_summaries.csv` - Review summaries with sentiment scores

### 3. Run Analysis Notebooks (Optional)

Run notebooks in order for data exploration and model development:

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

Notebooks:
- `01_exploration.ipynb` - Data exploration
- `02_feature_engineering.ipynb` - Feature creation
- `03_recommendation.ipynb` - Recommendation system
- `04_price_prediction.ipynb` - Price prediction models
- `05_review_nlp.ipynb` - Review sentiment analysis

## Demo Apps

### FastAPI Backend

Start the REST API server:

```bash
python -m uvicorn api.app:app --reload --host 127.0.0.1 --port 8000
```

Or use environment variables:

```bash
export TRAVEL_API_HOST=0.0.0.0
export TRAVEL_API_PORT=8000
python -m uvicorn api.app:app --reload
```

**API Endpoints:**

- `GET /health` - Health check
- `GET /meta/cities` - Get list of available cities
- `GET /meta/stats` - Get dataset statistics
- `POST /recommend` - Get hotel recommendations
  ```json
  {
    "city": "Goa",
    "budget": 6000.0,
    "min_rating": 4.0,
    "adults": 2,
    "limit": 10,
    "include_cancellation_risk": false
  }
  ```
- `POST /forecast/signal` - Get price forecast signal
  ```json
  {
    "city": "Goa",
    "hotel_class": 4.0,
    "check_in_date": "2024-12-25",
    "horizon_days": 7
  }
  ```
- `POST /risk/cancellation` - Get cancellation risk
  ```json
  {
    "hotel_type": "Resort Hotel",
    "market_segment": "Online TA",
    "lead_time": 30
  }
  ```

**API Documentation:**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Streamlit App

Run the interactive Streamlit demo (no backend required):

```bash
python -m streamlit run streamlit_app.py
```

Access at: http://localhost:8501

### React Frontend

1. Start the FastAPI backend (see above)
2. Open `web/index.html` in a browser
3. Configure API base URL (default: http://127.0.0.1:8000)

## Configuration

### Environment Variables

- `TRAVEL_DATA_DIR` - Override data directory path
- `TRAVEL_API_HOST` - API host (default: 127.0.0.1)
- `TRAVEL_API_PORT` - API port (default: 8000)

### Configuration File

Configuration is managed through `src/config.py`. Key settings:

- Default hotel class base: 3.5
- Price multiplier ranges
- Recommendation limits
- Forecast horizons

## Architecture Improvements

### Recent Enhancements

1. **Centralized Configuration**: All constants and paths in `src/config.py`
2. **Unified Data Loading**: Single source of truth in `src/data_loader.py`
3. **Code Reusability**: Shared utilities in `src/utils.py`
4. **Logging**: Structured logging with `src/logger.py`
5. **Error Handling**: Improved error handling in API endpoints
6. **Environment Support**: Environment variable configuration

### Code Quality

- Eliminated code duplication
- Improved maintainability
- Better error messages
- Type hints throughout
- Docstrings for all functions

## Development

### Project Structure Best Practices

- **Separation of Concerns**: Business logic separated from API/UI
- **DRY Principle**: No code duplication
- **Configuration Management**: Centralized config with env overrides
- **Logging**: Structured logging for debugging and monitoring

### Adding New Features

1. Add business logic in `src/` modules
2. Update API endpoints in `api/app.py` if needed
3. Add tests (when test framework is added)
4. Update documentation

## Dependencies

Key dependencies:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning
- `fastapi` - API framework
- `streamlit` - Interactive UI
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## Troubleshooting

### Data Loading Issues

- Ensure CSV files are in `data/` directory
- Check file permissions
- Verify CSV format matches expected schema

### API Connection Issues

- Check if FastAPI server is running
- Verify CORS settings if accessing from different origin
- Check logs for error messages

### Import Errors

- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check Python path includes project root

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
"# TravelIQ-" 
