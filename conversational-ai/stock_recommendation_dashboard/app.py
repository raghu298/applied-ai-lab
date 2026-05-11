"""Stock Recommendation Dashboard — Streamlit App."""

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import os
import csv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Stock Recommendation Dashboard", page_icon="📈", layout="wide")
st.title("📈 Stock Recommendation Dashboard")
st.markdown("Get stock data, news sentiment, and a BUY / HOLD / SELL recommendation.")

# --- Sentiment keywords ---
POSITIVE_WORDS = ["profit", "growth", "strong", "rise", "gain", "positive", "record", "beat", "expansion", "surge", "rally", "upgrade"]
NEGATIVE_WORDS = ["loss", "fall", "decline", "weak", "drop", "negative", "fraud", "debt", "penalty", "miss", "crash", "downgrade"]

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "data", "stock_history.csv")


def fetch_stock_data(symbol: str, period: str = "3mo") -> pd.DataFrame | None:
    """Fetch stock data using yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def get_stock_info(symbol: str) -> dict:
    """Get current stock info."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info
    except Exception:
        return {}


def fetch_news_headlines(symbol: str, max_headlines: int = 5) -> list[str]:
    """Fetch latest news headlines using Google News RSS."""
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
    url = f"https://news.google.com/rss/search?q={clean_symbol}+stock+news"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        items = soup.find_all("item", limit=max_headlines)
        return [item.title.text for item in items if item.title]
    except Exception:
        return []


def analyze_sentiment(headlines: list[str]) -> str:
    """Analyze news sentiment using keyword logic."""
    if not headlines:
        return "Neutral"

    combined = " ".join(headlines).lower()
    pos_count = sum(1 for w in POSITIVE_WORDS if w in combined)
    neg_count = sum(1 for w in NEGATIVE_WORDS if w in combined)

    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    return "Neutral"


def calculate_trend(current_price: float, ma_20: float) -> str:
    """Calculate stock trend based on 20-day MA."""
    if current_price > ma_20:
        return "Positive"
    elif current_price < ma_20:
        return "Negative"
    return "Neutral"


def get_recommendation(trend: str, sentiment: str) -> tuple[str, str]:
    """Generate recommendation based on trend and sentiment."""
    if trend == "Positive" and sentiment == "Positive":
        return "BUY 🟢", "Positive price trend with positive news sentiment."
    elif trend == "Positive" and sentiment == "Neutral":
        return "HOLD 🟡", "Positive price trend but neutral news sentiment."
    elif trend == "Negative" and sentiment == "Negative":
        return "SELL 🔴", "Negative price trend with negative news sentiment."
    else:
        return "HOLD 🟡", "Mixed signals — trend and sentiment don't align."


def save_to_history(record: dict):
    """Save result to stock_history.csv."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    file_exists = os.path.exists(HISTORY_FILE)
    fieldnames = ["timestamp", "symbol", "current_price", "prev_close", "ma_20", "trend", "sentiment", "recommendation", "reason"]

    with open(HISTORY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


# --- Input ---
st.header("🔎 Enter Stock Symbol")
col_input, col_examples = st.columns([2, 3])
with col_input:
    symbol = st.text_input("Stock Symbol", value="", placeholder="e.g., TCS.NS, AAPL, MSFT")
with col_examples:
    st.markdown("**Examples:** `TCS.NS` `INFY.NS` `RELIANCE.NS` `AAPL` `MSFT` `GOOGL`")

if st.button("📊 Analyze Stock"):
    if not symbol.strip():
        st.error("Please enter a stock symbol.")
    else:
        with st.spinner(f"Fetching data for **{symbol.upper()}**..."):
            df = fetch_stock_data(symbol.upper())

        if df is None or df.empty:
            st.error(f"Could not fetch data for '{symbol}'. Check the symbol and try again.")
        else:
            # Drop rows with NaN close prices (e.g., market closed)
            df = df.dropna(subset=["Close"])
            if df.empty:
                st.error("No valid price data available. Market may be closed.")
                st.stop()

            current_price = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) > 1 else current_price
            ma_20 = df["Close"].rolling(window=20).mean().iloc[-1]

            if pd.isna(ma_20):
                ma_20 = df["Close"].mean()  # fallback to simple average

            # Stock info
            info = get_stock_info(symbol.upper())

            # --- Price Metrics ---
            curr_sym = "₹" if symbol.upper().endswith((".NS", ".BO")) else "$"
            st.header("💰 Stock Overview")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Price", f"{curr_sym}{current_price:.2f}", f"{current_price - prev_close:+.2f}")
            with c2:
                st.metric("Previous Close", f"{curr_sym}{prev_close:.2f}")
            with c3:
                st.metric("20-Day Moving Avg", f"{curr_sym}{ma_20:.2f}")

            # --- Price Chart ---
            st.subheader("📉 Price Chart (Last 3 Months)")
            st.line_chart(df["Close"])

            # --- Last 5 rows ---
            st.subheader("📋 Recent Stock Data")
            st.dataframe(df.tail(5)[["Open", "High", "Low", "Close", "Volume"]], use_container_width=True)

            # --- News ---
            st.header("📰 Latest News Headlines")
            headlines = fetch_news_headlines(symbol.upper())
            if headlines:
                for i, headline in enumerate(headlines, 1):
                    st.markdown(f"**{i}.** {headline}")
            else:
                st.info("Could not fetch news headlines.")

            # --- Analysis ---
            sentiment = analyze_sentiment(headlines)
            trend = calculate_trend(current_price, ma_20)
            recommendation, reason = get_recommendation(trend, sentiment)

            st.header("🎯 Final Recommendation")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Trend", trend)
            with r2:
                st.metric("News Sentiment", sentiment)
            with r3:
                st.metric("Recommendation", recommendation)

            st.info(f"**Reason:** {reason}")

            # --- Save & Download ---
            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol.upper(),
                "current_price": f"{current_price:.2f}",
                "prev_close": f"{prev_close:.2f}",
                "ma_20": f"{ma_20:.2f}",
                "trend": trend,
                "sentiment": sentiment,
                "recommendation": recommendation.split()[0],
                "reason": reason,
            }
            save_to_history(record)

            result_df = pd.DataFrame([record])
            csv_data = result_df.to_csv(index=False)
            st.download_button("📥 Download Result", csv_data, f"{symbol}_result.csv", "text/csv")

# --- History ---
st.header("📜 Recommendation History")
if os.path.exists(HISTORY_FILE):
    hist_df = pd.read_csv(HISTORY_FILE)
    st.dataframe(hist_df, use_container_width=True)
else:
    st.info("No history yet. Analyze a stock to get started!")
