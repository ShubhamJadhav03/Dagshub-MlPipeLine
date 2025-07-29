import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow

# It's good practice to set these environment variables, but ensure they are managed securely,
# for example, using a .env file or environment management tools.
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/ShubhamJadhav03/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "ShubhamJadhav03"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8cacc2867ebf950241114286e5b19b5b8a0edf3c"


def hyperparameter_tuning(X_train, y_train, param_grid):
    """Performs grid search to find the best hyperparameters for RandomForest."""
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path, random_state, n_estimators, max_depth):
    """
    Trains a model, logs it with MLflow, and saves it to the specified path.
    Constructs absolute paths to ensure files are saved in the project root.
    """
    try:
        # --- PATH CORRECTION LOGIC ---
        # Get the project root directory (one level up from the 'src' folder)
        project_root = os.path.join(os.path.dirname(__file__), '..')

        # Construct full, absolute paths for the data and the output model
        full_data_path = os.path.join(project_root, data_path)
        full_model_path = os.path.join(project_root, model_path)
        # --- END OF CORRECTION ---

        print(f"Reading data from: {full_data_path}")
        data = pd.read_csv(full_data_path)
        
        X = data.drop(columns=["Outcome"])
        y = data['Outcome']

        mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

        with mlflow.start_run():
            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state)
            signature = infer_signature(X_train, y_train)

            # Define hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            # Perform hyperparameter tuning
            grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
            best_model = grid_search.best_estimator_

            # Predict and evaluate the model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")

            # Log parameters and metrics
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy)

            # Log confusion matrix and classification report as artifacts
            cm_text = str(confusion_matrix(y_test, y_pred))
            cr_text = classification_report(y_test, y_pred)
            mlflow.log_text(cm_text, "confusion_matrix.txt")
            mlflow.log_text(cr_text, "classification_report.txt")

            # Log the model
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            if tracking_url_type_store != 'file':
                mlflow.sklearn.log_model(best_model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(best_model, "model")

            # --- SAVE MODEL TO CORRECT PATH ---
            # Create the output directory if it doesn't exist, using the full path
            model_dir = os.path.dirname(full_model_path)
            os.makedirs(model_dir, exist_ok=True)

            # Save the trained model to the correct location using pickle
            with open(full_model_path, 'wb') as f:
                pickle.dump(best_model, f)

            print(f"Model saved to: {full_model_path}")

    except FileNotFoundError:
        print(f"Error: Input data file not found at {full_data_path}")
    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    params_file_path = os.path.join(os.path.dirname(__file__), '..', 'params.yaml')
    
    try:
        with open(params_file_path, 'r') as f:
            params = yaml.safe_load(f)["train"]
        
        # Run the training function with parameters from the YAML file
        train(
            data_path=params['data'],
            model_path=params['model'],
            random_state=params['random_state'],
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth']
        )
    except FileNotFoundError:
        print(f"Error: 'params.yaml' not found at {params_file_path}")
    except KeyError as e:
        print(f"Error: Missing key {e} in the 'train' section of params.yaml")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")