#!/bin/bash
# One-shot demo startup: stops everything, cleans old state, then opens
# four Terminal windows - Prefect server, pipelines, MLflow UI, and a
# spare terminal for api_access.py.
# Usage:  ./start_fresh.sh

PROJECT="/Users/raghunath/BITS_Pilani/3rd_sem/3rd_Sem_Labs/Cloud_Native/churn_pipeline"

echo "[1/3] Stopping any running services..."
pkill -f "prefect server" 2>/dev/null
pkill -f run_pipelines.py 2>/dev/null
pkill -f "mlflow ui" 2>/dev/null
sleep 3
pkill -f "mlflow.server" 2>/dev/null
pkill -f huey_consumer 2>/dev/null
sleep 2

echo "[2/3] Cleaning old state..."
# Prefect database (removes stale scheduled-run backlog and old runs)
rm -f ~/.prefect/prefect.db ~/.prefect/prefect.db-shm ~/.prefect/prefect.db-wal
# MLflow model artifact store (grows ~1 GB/hour while pipelines run)
rm -rf "$PROJECT/src/mlruns"
# MLflow tracking database so the run history starts fresh
rm -f "$PROJECT/mlflow.db"
# pipeline log so it shows only fresh runs
rm -f "$PROJECT/artifacts/pipeline.log"

echo "[3/3] Opening terminals..."
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd $PROJECT && source .venv/bin/activate && prefect server start"
    delay 12
    do script "cd $PROJECT/src && source ../.venv/bin/activate && PREFECT_API_URL=http://127.0.0.1:4200/api python run_pipelines.py"
    delay 2
    do script "cd $PROJECT && source .venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001"
    delay 2
    do script "cd $PROJECT/src && source ../.venv/bin/activate && clear && echo 'Terminal 4 ready. When runs appear, execute:  python api_access.py'"
end tell
EOF

echo ""
echo "Done. Four Terminal windows are starting."
echo "  Prefect dashboard : http://127.0.0.1:4200"
echo "  MLflow dashboard  : http://127.0.0.1:5001"
echo "First scheduled runs fire within 2 minutes."
echo "To stop everything later:  ./stop_all.sh"
