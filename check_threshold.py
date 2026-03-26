# check_threshold.py
import mlflow
import sys

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    run = mlflow.get_run(run_id)
    accuracy = run.data.metrics.get("accuracy", 0)
    
    if accuracy >= 0.85:
        sys.exit(0)
    else:
        sys.exit(1)
        
except Exception as e:
    sys.exit(1)