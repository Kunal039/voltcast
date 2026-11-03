"""List the versions of the registered forecaster in the MLflow model registry."""
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.MlflowClient()
for v in sorted(client.search_model_versions("name='voltcast-forecaster'"), key=lambda v: int(v.version)):
    run = client.get_run(v.run_id)
    print(f"{v.name}  version {v.version:>2}  status {v.status}  run {v.run_id[:12]}  "
          f"model={run.data.params.get('model')}  test_mae={run.data.metrics.get('test_mae', float('nan')):.4f}  "
          f"test_r2={run.data.metrics.get('test_r2', float('nan')):.4f}")
