import logging
import os
import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
import typer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Configure logging
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    datefmt="[%X]"
)

logger = logging.getLogger("model-training")
logger.setLevel("INFO")


def main(
    features_train = typer.Option(default=...),
    features_test = typer.Option(default=...),
    targets_train = typer.Option(default=...),
    targets_test = typer.Option(default=...),
) -> None:
    
    # Read the data from csv's into numpy arrays
    features_train_data = pd.read_csv(features_train).to_numpy()
    features_test_data = pd.read_csv(features_test).to_numpy()
    targets_train_data = pd.read_csv(targets_train).to_numpy()
    targets_test_data = pd.read_csv(targets_test).to_numpy()
    
    
    # Create a Support Vector Classifier (SVC) pipeline with StandardScaler
    svc = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='auto'))

    # Start a new MLflow run with a specific experiment ID
    mlflow.start_run(experiment_id="svc_model_training")
    
    # Fit the SVC model with training data
    svc.fit(features_train_data, targets_train_data)
    
    # Log and Register the trained SVC model using MLflow
    mlflow.sklearn.log_model(svc, 'model', registered_model_name="WeatherPrediction-Model")
    
    # End the MLflow run
    mlflow.end_run()

   

if __name__ == "__main__":
    typer.run(main)
