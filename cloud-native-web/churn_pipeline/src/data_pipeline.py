# Sub-Objective 1: Data pipeline for the telco customer churn problem.
# Steps: ingestion -> preprocessing -> EDA -> save processed data.
# The flow is scheduled every 2 minutes through Prefect (see run_pipelines.py).

import io
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_CSV = DATA_DIR / "telco_churn_raw.csv"
PROCESSED_CSV = DATA_DIR / "telco_churn_processed.csv"
LOG_FILE = ARTIFACT_DIR / "pipeline.log"


def setup_file_logger():
    logger = logging.getLogger("churn_pipeline")
    if logger.handlers:
        return logger

    ARTIFACT_DIR.mkdir(exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


FILE_LOGGER = setup_file_logger()


def log_pipeline_event(prefect_logger, message, *args):
    formatted_message = message % args if args else message
    prefect_logger.info(message, *args)
    FILE_LOGGER.info(formatted_message)


@task(name="data-ingestion", retries=2, retry_delay_seconds=10)
def ingest_data():
    logger = get_run_logger()
    DATA_DIR.mkdir(exist_ok=True)

    if RAW_CSV.exists():
        log_pipeline_event(logger, "Using cached dataset at %s", RAW_CSV)
        df = pd.read_csv(RAW_CSV)
    else:
        log_pipeline_event(logger, "Downloading dataset from %s", DATASET_URL)
        resp = requests.get(DATASET_URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        df.to_csv(RAW_CSV, index=False)

    log_pipeline_event(logger, "Ingested %d records, %d columns", len(df), df.shape[1])
    return df


@task(name="data-preprocessing")
def preprocess_data(df):
    logger = get_run_logger()

    log_pipeline_event(logger, "Data types:\n%s", df.dtypes.to_string())

    # TotalCharges comes in as object because of blank strings
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    log_pipeline_event(logger, "Summary statistics:\n%s", df.describe().round(2).to_string())

    missing = df.isna().sum()
    log_pipeline_event(
        logger,
        "Missing values:\n%s",
        missing[missing > 0].to_string() if missing.any() else "None",
    )

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            log_pipeline_event(logger, "Imputed missing values in %s with median %.2f", col, median)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    for i, col in enumerate(numeric_cols):
        df[col + "_norm"] = scaled[:, i]
    log_pipeline_event(logger, "Normalized columns: %s", list(numeric_cols))

    return df


@task(name="exploratory-data-analysis")
def run_eda(df):
    logger = get_run_logger()
    ARTIFACT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    base_numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    corr = df[base_numeric].corr()
    log_pipeline_event(logger, "Correlation matrix:\n%s", corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Correlation between numeric features")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "eda_correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # binning tenure into year groups
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    log_pipeline_event(logger, "Tenure bins:\n%s", df["tenure_group"].value_counts().to_string())

    # encoding
    df["Churn_encoded"] = (df["Churn"] == "Yes").astype(int)
    categorical_cols = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ]
    encoded = pd.get_dummies(df[categorical_cols], drop_first=True, dtype=int)
    df = pd.concat([df, encoded], axis=1)
    log_pipeline_event(
        logger,
        "Encoded %d categorical columns into %d features",
        len(categorical_cols),
        encoded.shape[1],
    )

    target_corr = (
        pd.concat([encoded, df["Churn_encoded"]], axis=1)
        .corr()["Churn_encoded"]
        .drop("Churn_encoded")
        .sort_values(key=abs, ascending=False)
    )
    log_pipeline_event(
        logger,
        "Features most correlated with churn:\n%s",
        target_corr.head(10).round(3).to_string(),
    )

    # feature importance
    feature_cols = base_numeric + list(encoded.columns)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(df[feature_cols], df["Churn_encoded"])
    importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    log_pipeline_event(logger, "Feature importance (top 10):\n%s", importance.head(10).round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(10).sort_values().plot.barh(ax=ax, color="#4878cf")
    ax.set_title("Top 10 feature importances")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "eda_feature_importance.png", dpi=120)
    plt.close(fig)

    # univariate plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#4878cf")
        ax.set_title("Distribution of " + col)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "eda_univariate_distributions.png", dpi=120)
    plt.close(fig)

    # bivariate plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0])
    axes[0].set_title("Churn by contract type")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("Monthly charges by churn")
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "eda_bivariate_analysis.png", dpi=120)
    plt.close(fig)

    log_pipeline_event(logger, "EDA charts saved to %s", ARTIFACT_DIR)
    return df


@task(name="save-processed-data")
def save_processed(df):
    logger = get_run_logger()
    df.to_csv(PROCESSED_CSV, index=False)
    log_pipeline_event(
        logger,
        "Processed dataset (%d rows, %d cols) written to %s",
        len(df),
        df.shape[1],
        PROCESSED_CSV,
    )
    return str(PROCESSED_CSV)


@flow(name="telco-churn-data-pipeline", log_prints=True)
def data_pipeline():
    raw = ingest_data()
    clean = preprocess_data(raw)
    enriched = run_eda(clean)
    return save_processed(enriched)


if __name__ == "__main__":
    data_pipeline()
