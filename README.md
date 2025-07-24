# mlops-assignment3

## Overview

This repository contains the solution for MLOps Assignment 3, focusing on building an end-to-end MLOps pipeline, including model development, Docker containerization, CI/CD with GitHub Actions, and model quantization.

## Project Structure

- `.github/workflows/ci.yml`: GitHub Actions workflow for CI/CD.
- `Dockerfile`: Defines the Docker image for the application.
- `requirements.txt`: Python dependencies.
- `train.py`: Script to train the scikit-learn Linear Regression model and save it.
- `predict.py`: Script to verify the Docker container by loading the model and making a prediction.
- `quantize.py`: Script to perform manual 8-bit quantization on the model parameters.
- `models/`: Directory to store the trained model and test data.
- `unquant_params.joblib`: Saved unquantized model parameters.
- `quant_params.joblib`: Saved manually quantized model parameters.
- `quantized_pytorch_model_state_dict.pth`: Saved PyTorch model state dict after quantization.

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Surbhi0616/mlops-assignment3.git](https://github.com/Surbhi0616/mlops-assignment3.git)
    cd mlops-assignment3
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv mlops3
    source mlops3/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## GitHub Actions CI/CD (docker_ci branch)

The `docker_ci` branch contains the CI/CD pipeline configured with GitHub Actions.
-   It builds a Docker image of the application.
-   Runs tests within the container using `predict.py`.
-   Pushes the Docker image to Docker Hub upon successful build and test.

## Model Quantization (quantization branch)

The `quantization` branch implements manual 8-bit quantization of the scikit-learn Linear Regression model's parameters.

1.  **Train the model (if not already done):**
    ```bash
    python train.py
    ```
2.  **Run the quantization script:**
    ```bash
    python quantize.py
    ```
    This script will print the R2 scores and model sizes for both the original and quantized models.

## Quantization Analysis

Here are the comparison results:

| Metric            | Original Sklearn Model | Quantized Model |
|-------------------|------------------------|-----------------|
| R^2 Score         |0.5757877060324588      |-4189.847842544245|
| Model Size (KB)   |0.40 KB                 |0.40 KB          |


## Docker Hub Link

[Link to your Docker Hub repository: `https://hub.docker.com/r/surbhi046/mlops-assignment3`](https://hub.docker.com/r/surbhi046/mlops-assignment3)

## GitHub Repository Link - https://github.com/Surbhi0616/mlops-assignment
