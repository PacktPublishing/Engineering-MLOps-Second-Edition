import logging
import os
import json
import mlflow.sklearn
import numpy as np
from sklearn.pipeline import Pipeline

def init():
    global model
    
    # Load the registered model from model path
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model")
    model = mlflow.sklearn.load_model(model_path)
    
    logging.info("Model loaded successfully")
    print("SVC model loaded")

def run(raw_data):
    try:
        # Parse the input data
        data = json.loads(raw_data)
        
        # Convert input data to numpy array
        input_data = np.array(data['data'])
        
        # Make predictions
        predictions = model.predict(input_data)
        
        # Create response
        response = {
            "predictions": predictions.tolist(),
            "model_info": {
                "model_type": "SVC",
                "pipeline_steps": [step.__class__.__name__ for step in model.steps]
            }
        }
        
        return response
        
    except Exception as e:
        error = str(e)
        return {"error": error}
