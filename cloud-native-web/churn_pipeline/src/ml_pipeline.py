# Sub-Objective 2: ML pipeline for churn prediction.
# Two algorithms are trained (Logistic Regression and Random Forest) on a
# 70/30 split. Metrics are logged to MLflow for monitoring (MLOps).

from pathlib import Path

import mlflow
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from data_pipeline import PROCESSED_CSV, data_pipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = "sqlite:///" + str(PROJECT_DIR / "mlflow.db")

BASE_NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
TARGET = "Churn_encoded"


@task(name="model-preparation")
def prepare_models():
    logger = get_run_logger()
    # Churn prediction is a binary classification problem, so two commonly
    # used classifiers were selected.
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    logger.info("Selected algorithms: %s", list(models.keys()))
    return models


@task(name="load-and-split")
def load_and_split():
    logger = get_run_logger()

    if not PROCESSED_CSV.exists():
        logger.info("Processed data not found, running the data pipeline first")
        data_pipeline()

    df = pd.read_csv(PROCESSED_CSV)
    encoded_cols = [c for c in df.columns
                    if "_" in c and c.split("_")[0] in (
                        "gender", "Partner", "Dependents", "PhoneService",
                        "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport",
                        "StreamingTV", "StreamingMovies", "Contract",
                        "PaperlessBilling", "PaymentMethod")]
    feature_cols = BASE_NUMERIC + encoded_cols

    X = df[feature_cols]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    logger.info("Train set: %d rows, Test set: %d rows (%d features)",
                len(X_train), len(X_test), X.shape[1])
    return X_train, X_test, y_train, y_test


@task(name="train-and-evaluate")
def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    logger = get_run_logger()

    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        mlflow.log_param("algorithm", name)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        logger.info("%s metrics: %s", name,
                    {k: round(v, 4) for k, v in metrics.items()})
    return metrics


@flow(name="telco-churn-ml-pipeline", log_prints=True)
def ml_pipeline():
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_DB)
    mlflow.set_experiment("telco-churn-prediction")

    models = prepare_models()
    X_train, X_test, y_train, y_test = load_and_split()

    results = {}
    for name, model in models.items():
        results[name] = train_and_evaluate(name, model, X_train, X_test,
                                           y_train, y_test)

    best = max(results, key=lambda k: results[k]["f1_score"])
    logger.info("Best model by F1 score: %s (%.4f)", best,
                results[best]["f1_score"])
    return results


if __name__ == "__main__":
    ml_pipeline()
