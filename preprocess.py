import pandas as pd
import sys
import os
import yaml

def preprocess_data(input_path, output_path):
    """
    Reads data from the input path, preprocesses it, and saves it to the output path.
    It constructs absolute paths from the project root to avoid creating folders
    inside the 'src' directory.
    """
    try:
        # --- PATH CORRECTION LOGIC ---
        # Get the project root directory (the parent of the 'src' folder)
        # __file__ is the path to the current script (e.g., /path/to/MLpipeline/src/preprocess.py)
        # os.path.dirname(__file__) is the directory of the script (e.g., /path/to/MLpipeline/src)
        # The '..' moves one level up to the project root (e.g., /path/to/MLpipeline)
        project_root = os.path.join(os.path.dirname(__file__), '..')

        # Construct the full, absolute path for both input and output files
        # by joining the project root with the relative paths from params.yaml.
        full_input_path = os.path.join(project_root, input_path)
        full_output_path = os.path.join(project_root, output_path)
        # --- END OF CORRECTION ---

        print(f"Reading data from: {full_input_path}")
        data = pd.read_csv(full_input_path)

        # --- PREPROCESSING STEPS WOULD GO HERE ---
        # For example:
        # data.dropna(inplace=True)
        # data['new_feature'] = data['existing_feature'] * 2
        print("Preprocessing complete.")

        # Create the output directory if it doesn't exist, using the full path
        output_dir = os.path.dirname(full_output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save the preprocessed data to the correct location
        data.to_csv(full_output_path, index=False)
        
        print(f"Data successfully preprocessed and saved to: {full_output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {full_input_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # This script expects the 'params.yaml' file to be in the project root
    params_file_path = os.path.join(os.path.dirname(__file__), '..', 'params.yaml')
    
    try:
        with open(params_file_path, 'r') as f:
            params = yaml.safe_load(f)
        
        # Get the input and output paths from the 'preprocess' section of the YAML file
        preprocess_params = params["preprocess"]
        input_file = preprocess_params["input"]
        output_file = preprocess_params["output"]

        # Run the main function
        preprocess_data(input_file, output_file)

    except FileNotFoundError:
        print(f"Error: 'params.yaml' not found at {params_file_path}")
    except KeyError:
        print("Error: 'preprocess', 'input', or 'output' key not found in params.yaml")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
