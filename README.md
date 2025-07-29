# Dagshub-MlPipeLine

End-to-End Machine Learning Pipeline with DVC and MLflow
This project demonstrates a complete MLOps pipeline for a classification task, using the Pima Indians Diabetes Database. The primary goal is to showcase best practices in experiment tracking, data versioning, and model management.

DagsHub Project Link: https://dagshub.com/ShubhamJadhav03/machinelearningpipeline

(This is a placeholder image. You should replace it with a screenshot of your MLflow dashboard on DagsHub, like the one you shared previously)

📋 Table of Contents
About the Project

Tech Stack

Project Structure

How to Run

Key Features

📖 About the Project
This repository contains the code and configuration for building a reproducible machine learning pipeline. The project predicts the onset of diabetes based on diagnostic measures. It goes beyond just training a model by incorporating key MLOps tools to create a robust and version-controlled workflow.

🛠️ Tech Stack
Language: Python

ML Framework: Scikit-learn

Experiment Tracking: MLflow

Data & Model Versioning: DVC (Data Version Control)

Platform: DagsHub (for hosting the repository and integrated MLflow server)

Configuration: YAML

📂 Project Structure
.
├── data/                     # Data files (tracked by DVC)

│   └── diabetes.csv

├── models/                   # Saved model artifacts (tracked by DVC)

│   └── model.pkl

├── src/                      # Source code for the pipeline


│   └── train.py              # Main training script

├── .dvc/                     # DVC metadata files

├── .gitignore

├── dvc.yaml                  # DVC pipeline definition

├── params.yaml               # Parameters for the pipeline (hyperparameters, etc.)

├── requirements.txt          # Python dependencies

└── README.md

🚀 How to Run
To reproduce this project, follow these steps:

1. Clone the repository:

git clone <your-github-repo-url>
cd <repository-name>

2. Install dependencies:
It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt

3. Pull the data and models versioned with DVC:
This will download the diabetes.csv dataset and any pre-trained models from your DagsHub remote storage.

dvc pull

4. Run the training pipeline:
This command executes the stages defined in dvc.yaml, which runs the train.py script. MLflow will automatically track the experiment on your DagsHub server.

dvc repro

After the run is complete, you can view the newly logged experiment, including metrics, parameters, and model artifacts, on your DagsHub MLflow server.

✨ Key Features
Data Versioning with DVC: The dataset is tracked by DVC, not Git. This keeps the repository lightweight and ensures that every experiment is tied to a specific version of the data.

Experiment Tracking with MLflow: Every training run is logged as an MLflow experiment. This includes:

Hyperparameters used for the run.

Performance metrics (e.g., accuracy).

The trained model as an artifact.

Confusion matrix and classification report.

Reproducibility: By combining Git (for code), DVC (for data), and MLflow (for parameters), any past experiment can be perfectly reproduced.

Centralized Platform with DagsHub: DagsHub provides a single interface to view code, data, models, and MLflow experiments, streamlining the entire MLOps workflow.
