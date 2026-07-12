#!/bin/bash
# Stops the Prefect server, the pipelines, and the MLflow UI.
# Escalates to force-kill if anything is still alive after 5 seconds.
# Usage:  ./stop_all.sh

pkill -f "prefect server" 2>/dev/null
pkill -f run_pipelines.py 2>/dev/null
pkill -f "mlflow ui" 2>/dev/null
sleep 3
pkill -f "mlflow.server" 2>/dev/null
pkill -f huey_consumer 2>/dev/null
sleep 2

# force-kill anything that ignored the first signal
if pgrep -f "prefect server|run_pipelines.py|mlflow ui|mlflow.server|huey_consumer" > /dev/null 2>&1; then
    pkill -9 -f "prefect server" 2>/dev/null
    pkill -9 -f run_pipelines.py 2>/dev/null
    pkill -9 -f "mlflow ui" 2>/dev/null
    pkill -9 -f "mlflow.server" 2>/dev/null
    pkill -9 -f huey_consumer 2>/dev/null
    sleep 1
fi

if pgrep -fl "prefect server|run_pipelines.py|mlflow ui|mlflow.server|huey_consumer" 2>/dev/null; then
    echo "Warning: something is still running (listed above)."
else
    echo "All services stopped."
fi
