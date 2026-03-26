import mlflow
import sys
import os

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    print("ERROR: MLFLOW_TRACKING_URI not set!")
    sys.exit(1)

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