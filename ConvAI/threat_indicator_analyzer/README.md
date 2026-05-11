# AI Threat Indicator Analyzer

A Streamlit-based application that analyzes threat indicators and classifies them as **Safe**, **Suspicious**, **Malicious**, or **Critical**.

## Features

- Accepts multiple indicator types: IP Address, URL, Domain, CVE ID, File Hash, Email Address
- Upload and parse log files / threat reports to auto-extract indicators
- Rule-based threat analysis with risk scoring (0–100)
- Optional LLM-powered analysis via OpenRouter API
- Results logged to CSV for audit trail
- Clean Streamlit dashboard with download functionality

## Project Structure

```
threat_indicator_analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # API keys (not committed)
├── .env.example            # Template for .env
├── PROJECT_RULES.md        # Development rules
├── README.md               # This file
├── modules/
│   ├── validator.py        # Input validation
│   ├── threat_analyzer.py  # Rule-based & LLM analysis
│   ├── report_parser.py    # File parsing & indicator extraction
│   ├── risk_scorer.py      # Risk score utilities
│   └── logger.py           # CSV logging
├── data/
│   └── sample_threat_inputs.json
└── logs/
    └── analysis_history.csv
```

## Setup & Run

```bash
# 1. Navigate to project folder
cd threat_indicator_analyzer

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the app
streamlit run app.py
```

## Sample Input

| Type         | Example                                     |
|--------------|---------------------------------------------|
| IP Address   | 185.220.101.45                               |
| URL          | http://secure-bank-login.verify-now.com      |
| Domain       | fake-update-security.com                     |
| CVE ID       | CVE-2023-34362                               |
| File Hash    | 44d88612fea8a8f36de82e1278abb02f             |
| Email        | admin@paypal-security-alert.com              |
