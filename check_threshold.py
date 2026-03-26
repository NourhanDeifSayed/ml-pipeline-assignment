import mlflow
import sys
import os

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    print("ERROR: DAGSHUB_TOKEN not set!")
    sys.exit(1)

os.environ["MLFLOW_TRACKING_USERNAME"] = "NourhanDeifSayed"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

tracking_uri = "https://dagshub.com/NourhanDeifSayed/ml-pipeline-assignment.mlflow"
mlflow.set_tracking_uri(tracking_uri)

run = mlflow.get_run(run_id)
accuracy = run.data.metrics.get("accuracy")

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy:.4f}")

if accuracy >= 0.85:
    print("PASSED - Accuracy meets threshold. Proceeding to deployment.")
    sys.exit(0)
else:
    print(f"FAILED - Accuracy {accuracy:.4f} is below 0.85 threshold.")
    sys.exit(1)