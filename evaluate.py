import os
import pandas as pd
import pickle
import yaml
import mlflow
from sklearn.metrics import accuracy_score
from urllib.parse import urlparse

# Set MLflow tracking environment variables
# For production, consider using a more secure method like .env files
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/ShubhamJadhav03/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "ShubhamJadhav03"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8cacc2867ebf950241114286e5b19b5b8a0edf3c"


def evaluate(data_path, model_path):
    """
    Loads a trained model and evaluates its performance on the provided data.
    Constructs absolute paths to ensure the model and data are found correctly.
    """
    try:
        # --- PATH CORRECTION LOGIC ---
        # Get the project root directory (one level up from the 'src' folder)
        project_root = os.path.join(os.path.dirname(__file__), '..')

        # Construct full, absolute paths for the data and the model file
        full_data_path = os.path.join(project_root, data_path)
        full_model_path = os.path.join(project_root, model_path)
        # --- END OF CORRECTION ---

        print(f"Reading data for evaluation from: {full_data_path}")
        data = pd.read_csv(full_data_path)
        X = data.drop(columns=["Outcome"])
        y = data['Outcome']

        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

        print(f"Loading model from: {full_model_path}")
        # --- FIX IS HERE ---
        # Load the model from the disk using the correct full path
        with open(full_model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Make predictions and calculate accuracy
        predictions = model.predict(X)
        accuracy = accuracy_score(y, predictions)

        # Start an MLflow run to log evaluation metrics
        with mlflow.start_run():
            mlflow.log_metric("evaluation_accuracy", accuracy)
            print(f"Logged evaluation accuracy to MLflow: {accuracy}")

    except FileNotFoundError:
        print(f"Error: Model or data file not found.")
        print(f"Attempted to load model from: {full_model_path}")
        print(f"Attempted to load data from: {full_data_path}")
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")


if __name__ == "__main__":
    # Load parameters from the 'evaluate' section of params.yaml
    params_file_path = os.path.join(os.path.dirname(__file__), '..', 'params.yaml')
    
    try:
        with open(params_file_path, 'r') as f:
            # Note: Assuming evaluation uses the same data and model paths as training
            # You might want to create a separate 'evaluate' section in params.yaml
            params = yaml.safe_load(f)["train"]
        
        # Run the evaluation function
        evaluate(data_path=params["data"], model_path=params["model"])

    except FileNotFoundError:
        print(f"Error: 'params.yaml' not found at {params_file_path}")
    except KeyError as e:
        print(f"Error: Missing key {e} in the 'train' section of params.yaml")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
