# Sub-Objective 3: Access application details through Prefect's built-in
# REST API and display them. Requires the Prefect server to be running
# (prefect server start) and the pipelines to be served (run_pipelines.py).

import requests

API_URL = "http://127.0.0.1:4200/api"


def post(endpoint, body=None):
    resp = requests.post(API_URL + endpoint, json=body or {}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get(endpoint):
    resp = requests.get(API_URL + endpoint, timeout=10)
    resp.raise_for_status()
    return resp.json()


def show_health():
    print("=" * 60)
    print("1. Server health (GET /health)")
    print("=" * 60)
    print("Server healthy:", get("/health"))
    version = get("/admin/version")
    print("Prefect version:", version)


def show_flows():
    print("=" * 60)
    print("2. Registered flows (POST /flows/filter)")
    print("=" * 60)
    for f in post("/flows/filter"):
        print(f"  Flow: {f['name']}  (id: {f['id']})")


def show_deployments():
    print("=" * 60)
    print("3. Deployments and schedules (POST /deployments/filter)")
    print("=" * 60)
    for d in post("/deployments/filter"):
        schedules = d.get("schedules") or []
        interval = schedules[0]["schedule"].get("interval") if schedules else None
        print(f"  Deployment: {d['name']}")
        print(f"    status: {d['status']}, paused: {d['paused']}, "
              f"interval: {interval}s, tags: {d['tags']}")


def show_recent_runs(limit=10):
    print("=" * 60)
    print(f"4. Last {limit} executed flow runs (POST /flow_runs/filter)")
    print("=" * 60)
    body = {
        "limit": limit,
        "sort": "START_TIME_DESC",
        "flow_runs": {"state": {"type": {"any_": ["COMPLETED", "RUNNING", "FAILED"]}}},
    }
    for r in post("/flow_runs/filter", body):
        duration = r.get("total_run_time")
        print(f"  Run: {r['name']:<25} state: {r['state_type']:<10} "
              f"start: {r['start_time']}  duration: {duration}s")


def show_recent_logs(limit=10):
    print("=" * 60)
    print(f"5. Last {limit} log records (POST /logs/filter)")
    print("=" * 60)
    body = {"limit": limit, "sort": "TIMESTAMP_DESC"}
    for entry in post("/logs/filter", body):
        message = entry["message"].splitlines()[0][:80]
        print(f"  [{entry['timestamp']}] {message}")


def show_run_counts():
    print("=" * 60)
    print("6. Flow run counts by state (POST /flow_runs/count)")
    print("=" * 60)
    for state in ["COMPLETED", "FAILED", "RUNNING", "SCHEDULED"]:
        body = {"flow_runs": {"state": {"type": {"any_": [state]}}}}
        count = post("/flow_runs/count", body)
        print(f"  {state:<10}: {count}")


if __name__ == "__main__":
    show_health()
    show_flows()
    show_deployments()
    show_recent_runs()
    show_recent_logs()
    show_run_counts()
