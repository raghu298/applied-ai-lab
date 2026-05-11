# ConvAI — AI App Experiments

## Experiments

### 1. AI Threat Indicator Analyzer (`threat_indicator_analyzer/`)
Streamlit app that classifies cybersecurity threat indicators (IPs, URLs, domains, CVEs, hashes, emails) as Safe / Suspicious / Malicious / Critical.

### 2. Stock Recommendation Dashboard (`stock_recommendation_dashboard/`)
Streamlit app that fetches stock data, scrapes news, analyzes sentiment, and gives BUY / HOLD / SELL recommendations.

## Quick Start

```bash
cd <experiment_folder>
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
- Python, Streamlit, Pandas
- OpenRouter / Gemini API (optional LLM)
- yfinance, BeautifulSoup
