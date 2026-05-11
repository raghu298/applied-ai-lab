# Stock Recommendation Dashboard

A Streamlit app that fetches stock data, scrapes news, analyzes trends & sentiment, and recommends **BUY / HOLD / SELL**.

## Features

- Fetch 3-month stock data via **yfinance**
- Display current price, previous close, and 20-day moving average
- Interactive price chart
- Scrape latest news headlines from Google News RSS
- Keyword-based sentiment analysis (Positive / Neutral / Negative)
- Trend analysis using 20-day moving average
- Final recommendation: BUY / HOLD / SELL with reason
- Save results to CSV with download button

## Project Structure

```
stock_recommendation_dashboard/
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── .env.example         # Template for .env
├── .gitignore
├── README.md            # This file
└── data/
    └── stock_history.csv  # Auto-generated history
```

## Setup & Run

```bash
# 1. Navigate to project folder
cd stock_recommendation_dashboard

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

## Supported Stock Symbols

| Symbol       | Company          |
|-------------|------------------|
| TCS.NS      | TCS (NSE)        |
| INFY.NS     | Infosys (NSE)    |
| RELIANCE.NS | Reliance (NSE)   |
| AAPL        | Apple (NASDAQ)   |
| MSFT        | Microsoft        |
| GOOGL       | Alphabet/Google  |
