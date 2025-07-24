# train.py
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_and_save_model():
    # Load the California Housing dataset
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)

    # Split the data into training and testing sets
    # Use a fixed random_state for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a directory to save the model if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the trained model using joblib
    model_path = 'models/linear_regression_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

    # Save test data for later use in predict.py and quantize.py
    test_data_path = 'models/test_data.joblib'
    joblib.dump((X_test, y_test), test_data_path)
    print(f"Test data saved to {test_data_path}")

if __name__ == "__main__":
    train_and_save_model()
