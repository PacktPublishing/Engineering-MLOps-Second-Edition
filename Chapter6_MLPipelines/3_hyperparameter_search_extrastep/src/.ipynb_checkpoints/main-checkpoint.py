import logging
import os
import mlflow
from pathlib import Path
import numpy as np
import pandas as pd
import typer
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV



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
    
    print("printing paths")
    print(features_train)
    print(features_test)
    print(targets_train)
    print(targets_test)
    

    # Read the data from csv's into Pandas DataFrames
    features_train_data = pd.read_csv(features_train).to_numpy()
    features_test_data = pd.read_csv(features_test).to_numpy()
    targets_train_data = pd.read_csv(targets_train).to_numpy()
    targets_test_data = pd.read_csv(targets_test).to_numpy()
    
    print(features_train_data.shape)
    print(targets_train_data.shape)
    print(pd.read_csv(features_train).columns)
    print(pd.read_csv(targets_train).columns)
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
     
    svc = svm.SVC()
    svc_grid = GridSearchCV(svc, parameters)
    svc_grid.fit(features_train_data, targets_train_data)
    print(svc_grid.get_params(deep=True))   
    
    

if __name__ == "__main__":
    typer.run(main)