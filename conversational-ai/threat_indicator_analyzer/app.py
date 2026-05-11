"""AI Threat Indicator Analyzer — Streamlit Dashboard."""

import streamlit as st
import pandas as pd
import json
import os
from dotenv import load_dotenv

from modules.validator import validate_indicator
from modules.threat_analyzer import analyze_indicator, analyze_with_llm
from modules.report_parser import parse_uploaded_file, extract_indicators_from_text
from modules.risk_scorer import get_risk_color, get_classification_color
from modules.logger import log_result, get_history

load_dotenv()

st.set_page_config(page_title="AI Threat Indicator Analyzer", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Threat Indicator Analyzer")
st.markdown("Analyze threat indicators and classify them as **Safe**, **Suspicious**, **Malicious**, or **Critical**.")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
use_llm = st.sidebar.checkbox(
    "Use LLM Analysis (requires OpenRouter API key)",
    value=False,
    help="Enable AI-powered analysis via OpenRouter. Falls back to rule-based if unavailable.",
)

# --- Input Section ---
st.header("📥 Input")
indicator_type = st.selectbox(
    "Select Indicator Type",
    ["IP Address", "URL", "Domain", "CVE ID", "File Hash", "Email Address", "Log File", "Threat Report"],
)

results_to_display = []

if indicator_type in ("Log File", "Threat Report"):
    uploaded_file = st.file_uploader(
        f"Upload a {indicator_type}",
        type=["txt", "csv", "log"],
        help="Supported formats: .txt, .csv, .log",
    )
    if st.button("🔍 Analyze File") and uploaded_file:
        with st.spinner("Parsing and analyzing file..."):
            content = parse_uploaded_file(uploaded_file)
            indicators = extract_indicators_from_text(content)
            if not indicators:
                st.warning("No threat indicators found in the uploaded file.")
            else:
                st.success(f"Found **{len(indicators)}** indicator(s) in the file.")
                for ind in indicators:
                    if use_llm:
                        result = analyze_with_llm(ind["type"], ind["value"])
                    else:
                        result = None
                    if result is None:
                        result = analyze_indicator(ind["type"], ind["value"])
                    log_result(result)
                    results_to_display.append(result)
else:
    indicator_value = st.text_input(
        f"Enter {indicator_type}",
        placeholder=f"e.g., {'185.220.101.45' if indicator_type == 'IP Address' else 'example value'}",
    )
    if st.button("🔍 Analyze"):
        if not indicator_value.strip():
            st.error("Please enter a value to analyze.")
        elif not validate_indicator(indicator_type, indicator_value):
            st.error(f"❌ Invalid {indicator_type} format. Please check your input.")
        else:
            with st.spinner("Analyzing..."):
                if use_llm:
                    result = analyze_with_llm(indicator_type, indicator_value)
                else:
                    result = None
                if result is None:
                    result = analyze_indicator(indicator_type, indicator_value)
                log_result(result)
                results_to_display.append(result)

# --- Results Section ---
if results_to_display:
    st.header("📊 Analysis Results")
    for result in results_to_display:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Indicator Type", result["indicator_type"])
        with col2:
            st.metric("Risk Score", f"{get_risk_color(result['risk_score'])} {result['risk_score']}/100")
        with col3:
            st.metric("Classification", get_classification_color(result["classification"]))

        st.markdown(f"**Indicator:** `{result['indicator_value']}`")
        st.markdown(f"**Reason:** {result['reason']}")
        st.markdown(f"**Recommended Action:** {result['recommended_action']}")

        # JSON output
        with st.expander("📋 View JSON Output"):
            st.json(result)
        st.divider()

# --- History Section ---
st.header("📜 Analysis History")
history = get_history()
if history:
    df = pd.DataFrame(history)
    st.dataframe(df, use_container_width=True)
    csv_data = df.to_csv(index=False)
    st.download_button("📥 Download History CSV", csv_data, "analysis_history.csv", "text/csv")
else:
    st.info("No analysis history yet. Analyze an indicator to get started!")

# --- Sample Data ---
with st.expander("📂 View Sample Threat Inputs"):
    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_threat_inputs.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r") as f:
            samples = json.load(f)
        st.json(samples)
    else:
        st.info("Sample data file not found.")
