# DataOps scheduling: serves both pipelines as Prefect deployments that run
# every 2 minutes. Run "prefect server start" in another terminal first,
# then run this script. Progress and logs show up on the Prefect dashboard
# at http://127.0.0.1:4200.

from prefect import serve

from data_pipeline import data_pipeline
from ml_pipeline import ml_pipeline

if __name__ == "__main__":
    data_deployment = data_pipeline.to_deployment(
        name="data-pipeline-every-2-min",
        interval=120,
        tags=["dataops", "assignment-1"],
    )
    ml_deployment = ml_pipeline.to_deployment(
        name="ml-pipeline-every-2-min",
        interval=120,
        tags=["mlops", "assignment-1"],
    )
    serve(data_deployment, ml_deployment)
