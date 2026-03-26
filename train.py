import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import os

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

if random.random() < 0.3:
    accuracy = accuracy * 0.7

dagshub_token = os.getenv("DAGSHUB_TOKEN")
if not dagshub_token:
    print("ERROR: DAGSHUB_TOKEN not set!")
    exit(1)

tracking_uri = f"https://{dagshub_token}@dagshub.com/NourhanDeifSayed/ml-pipeline-assignment.mlflow"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("ml-pipeline-assignment")

with mlflow.start_run() as run:
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")