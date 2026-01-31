# Current Features List

## Core Recommendation Features

### 1. **Hotel Recommendations**
- **Explainable AI Recommendations**: Get hotel recommendations with transparent reasoning
- **Multi-criteria Filtering**:
  - City/location filtering
  - Budget constraints
  - Minimum rating requirements
  - Number of adults/children
  - Hotel type preferences
  - Meal preferences
- **Scoring System**: Weighted scoring based on:
  - Price match (30% weight)
  - Sentiment score (30% weight)
  - Average rating (25% weight)
  - Review volume (15% weight)
- **Explainable Results**: Each recommendation includes:
  - Recommendation score
  - Reason for recommendation
  - Pros and cons from reviews
  - Price confidence level
  - Sentiment analysis

### 2. **Baseline Recommendations**
- Simple filtering-based recommendations
- Sort by price, special requests, lead time

### 3. **Review-Based Recommendations**
- Recommendations enriched with review insights
- Sentiment analysis integration

## Price Analysis Features

### 4. **Price Forecasting**
- **Forecast Signal**: Predict price trends for booking timing advice
  - UP/DOWN/STABLE trend prediction
  - Expected percentage change
  - Confidence scoring
  - Booking advice (BOOK_NOW/WAIT/STABLE/FLEXIBLE_BOOKING)
  - Volatility assessment (LOW/MEDIUM/HIGH)
- **Multi-horizon Forecasting**: Predict prices for multiple future dates
- **City and Hotel Class Scaling**: Adjusts predictions based on location and hotel quality

### 5. **Price Fairness Analysis**
- Compare current prices against historical averages
- Categorization: "Great deal", "Fair price", "Slightly high", "Overpriced"
- Percentage difference calculation
- Hotel class adjustments
- Monthly seasonality consideration

### 6. **Best Booking Window**
- Recommend optimal lead time for booking
- Identifies lead time buckets with lowest median prices
- Confidence scoring based on sample size
- Month-specific recommendations

### 7. **Price Timing Signal**
- Predict whether to book now or wait
- Compares current price vs. future price prediction
- Decision output: BOOK_NOW/WAIT/STABLE
- Uses machine learning model (Ridge regression)

## Risk Analysis Features

### 8. **Cancellation Risk Analysis**
- Predict cancellation probability based on:
  - Hotel type
  - Market segment
  - Lead time (bucketed: 0-7, 8-30, 31-90, 91-180, 181+ days)
- Provides cancellation rate percentage
- Sample size information for reliability
- Actionable advice messages

### 9. **Overbooking Risk Analysis**
- Evaluate room reassignment risk
- Analyze waitlist probability
- Risk score calculation (0.0-1.0)
- Risk levels: low/medium/high
- Month-specific analysis
- User-specific adjustments (repeated guests, cancellation history)

## Data & Infrastructure Features

### 10. **Centralized Data Loading**
- Unified data loading functions
- Hotel bookings data loading
- Offerings/hotel metadata loading
- Review summaries loading
- Hotel candidate preparation with price estimates

### 11. **Configuration Management**
- Centralized configuration in `src/config.py`
- Environment variable support (`TRAVEL_DATA_DIR`, `TRAVEL_API_HOST`, `TRAVEL_API_PORT`)
- Configurable defaults (budget, ratings, limits)
- Path management

### 12. **Logging System**
- Structured logging with timestamps
- Console and file logging support
- Error logging with stack traces
- Configurable log levels

### 13. **Error Handling**
- Comprehensive error handling in API endpoints
- User-friendly error messages
- HTTP error responses
- Error logging for debugging

## API Endpoints

### 14. **REST API (FastAPI)**
- `GET /health` - Health check endpoint
- `GET /meta/cities` - Get list of available cities
- `GET /meta/stats` - Get dataset statistics
- `POST /recommend` - Get hotel recommendations
  - Supports all recommendation filters
  - Optional cancellation risk inclusion
- `POST /forecast/signal` - Get price forecast signal
- `POST /risk/cancellation` - Get cancellation risk analysis
- **Auto-generated Documentation**:
  - Swagger UI at `/docs`
  - ReDoc at `/redoc`
- **CORS Support**: Enabled for cross-origin requests

## User Interfaces

### 15. **Streamlit Application**
- Interactive web interface
- No backend required (runs independently)
- Filters: City, Budget, Min Rating, Adults, Results Limit
- Real-time recommendations display
- Cached data loading for performance

### 16. **React Frontend**
- Static HTML/JavaScript application
- Connects to FastAPI backend
- City autocomplete
- Booking advice integration
- Forecast display

## Data Processing Features

### 17. **Hotel Candidate Preparation**
- Automatic price estimation based on:
  - Global ADR baseline
  - Hotel class multipliers
  - City location multipliers
- Price confidence levels (low/medium)
- Price range estimation (low/high bounds)

### 18. **Review Sentiment Integration**
- Sentiment scores from reviews
- Average ratings
- Review volume
- Pros and cons extraction

## Machine Learning Features

### 19. **Price Prediction Model**
- Ridge regression model for ADR prediction
- Feature engineering:
  - Hotel type
  - Lead time
  - Date features (year, month, week, day)
  - Guest information
  - Market segment
  - Distribution channel
  - And more (30+ features)
- Model training and evaluation
- Prediction capabilities

## Utility Features

### 20. **Data Utilities**
- Locality parsing from address strings
- Safe numeric conversion
- Project root detection
- Date/time parsing and handling

### 21. **Statistical Analysis**
- Linear trend fitting
- Volatility calculation (coefficient of variation)
- Statistical aggregations (mean, median, percentiles)

## Development Features

### 22. **Jupyter Notebooks**
- `01_exploration.ipynb` - Data exploration
- `02_feature_engineering.ipynb` - Feature creation
- `03_recommendation.ipynb` - Recommendation system development
- `04_price_prediction.ipynb` - Price prediction models
- `05_review_nlp.ipynb` - Review sentiment analysis

## Summary by Category

**Core Features**: 9 features (Recommendations, Price Analysis, Risk Analysis)
**API & Infrastructure**: 5 features (API endpoints, Data loading, Config, Logging, Error handling)
**User Interfaces**: 2 features (Streamlit, React)
**Machine Learning**: 1 feature (Price prediction model)
**Utilities & Tools**: 5 features (Data utilities, Statistical analysis, Notebooks)

**Total: 22 Major Features**

---

*Last Updated: Based on current codebase analysis*
