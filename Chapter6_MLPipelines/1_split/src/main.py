import logging
import os
from pathlib import Path
import typer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Configure logging
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    datefmt="[%X]"
)

logger = logging.getLogger("split-data")
logger.setLevel("INFO")

def main(
    input_data: str = typer.Option(default=..., help="File path to the input dataset."),
    random_state: int = typer.Option(default=...),
    test_size: float = typer.Option(default=...),
    features_train = typer.Option(default=..., help="File path to the features train dataset."),
    features_test = typer.Option(default=..., help="File path to the features test dataset."),
    targets_train = typer.Option(default=..., help="File path to the targets train dataset."),
    targets_test = typer.Option(default=..., help="File path to the targets test dataset."),
) -> None:
    logger.info("Starting train/validation/test split job.")
    

    
    # Read the features and targets data
    input_data = pd.read_csv(input_data)

    print("printing shapes")
    print(input_data.shape)
    print(input_data.columns)
    
    target_columns = ['Future_weather_condition']
    features_columns = ['Temperature_C', 'Humidity', 'Wind_speed_kmph', 'Wind_bearing_degrees', 'Visibility_km', 'Pressure_millibars', 'Current_weather_condition']
    
    features_data = input_data[features_columns]
    targets_data = input_data[target_columns]
    

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_data,
        targets_data,
        test_size=test_size,
        random_state=random_state
    )



    # Saving features and targets 
    X_train.to_csv(features_train,index=False)
    y_train.to_csv(targets_train,index=False)
    X_test.to_csv(features_test,index=False)
    y_test.to_csv(targets_test,index=False)

    
    logger.info("Train/test split job completed.")

if __name__ == "__main__":
    typer.run(main)

    
 