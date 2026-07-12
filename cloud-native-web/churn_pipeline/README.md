# Telco Customer Churn Prediction - Cloud Native Data Science Application

Assignment 1 for AIMLCZG549 (API-driven Cloud Native Solutions).

## Business problem

Telecom operators lose recurring revenue whenever a customer cancels their
subscription. The goal of this application is to predict which customers are
likely to churn using their demographic, service and billing attributes, so
the business can target them with retention offers.

Dataset: IBM Telco Customer Churn (available on Kaggle), 7,043 records and
21 columns.

## Architecture

- Prefect: workflow orchestration, 2-minute scheduling, logging, dashboard
  (Sub-Objectives 1 and 3)
- MLflow: experiment tracking and model metric monitoring (Sub-Objective 2)
- scikit-learn / pandas / seaborn: preprocessing, EDA and modelling

```
churn_pipeline/
  src/
    data_pipeline.py   ingestion, preprocessing, EDA (Prefect flow)
    ml_pipeline.py     model training, evaluation, MLflow logging (Prefect flow)
    run_pipelines.py   serves both flows on a 2-minute schedule
    api_access.py      retrieves application details via the Prefect REST API
  data/                raw and processed CSVs
  artifacts/           EDA charts (PNG)
  mlflow.db            MLflow tracking database
```

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## How to run

Use three terminals from the `churn_pipeline` directory.

Terminal 1 - Prefect server (dashboard at http://127.0.0.1:4200):

```
.venv/bin/prefect server start
```

Terminal 2 - serve both pipelines on a 2-minute schedule:

```
cd src
PREFECT_API_URL=http://127.0.0.1:4200/api ../.venv/bin/python run_pipelines.py
```

Terminal 3 - MLflow dashboard (at http://127.0.0.1:5001):

```
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

(Port 5001 is used because macOS AirPlay Receiver occupies port 5000.)

To run either pipeline once manually:

```
cd src
../.venv/bin/python data_pipeline.py
../.venv/bin/python ml_pipeline.py
```

Sub-Objective 3 (application details via the built-in REST API):

```
cd src
../.venv/bin/python api_access.py
```

## What each sub-objective maps to

1. Data pipeline (data_pipeline.py)
   - 1.1 Business understanding: churn prediction (above)
   - 1.2 Ingestion: dataset downloaded from a public repository and cached
   - 1.3 Preprocessing: dtypes, summary statistics, missing value check,
     median imputation, min-max normalization
   - 1.4 EDA: correlation matrix and heatmap, correlation of encoded
     categorical features with the target, tenure binning, one-hot encoding,
     random forest feature importance, univariate histograms, bivariate
     count/box plots
   - 1.5 DataOps: flow served every 120 seconds, all steps logged, visible
     on the Prefect dashboard

2. ML pipeline (ml_pipeline.py)
   - 2.1 Model preparation: Logistic Regression and Random Forest
   - 2.2 Training: 70/30 stratified train-test split
   - 2.3 Evaluation: accuracy on the test set
   - 2.4 MLOps: accuracy, precision, recall, F1 and ROC-AUC logged to MLflow
     on every scheduled run

3. API access (api_access.py) - uses Prefect's built-in REST API to display:
   server health and version, registered flows, deployments with schedules,
   recent flow runs, work pools, and flow run counts by state
