# predict.py
import joblib
import os
from sklearn.metrics import r2_score
import pandas as pd # Ensure pandas is imported if X_test is a DataFrame

def verify_model_prediction():
    try:
        # Load the saved scikit-learn model
        model_path = 'models/linear_regression_model.joblib'
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

        # Load the test data
        test_data_path = 'models/test_data.joblib'
        X_test, y_test = joblib.load(test_data_path)
        print(f"Test data loaded from {test_data_path}")

        # Ensure X_test is a DataFrame if it was saved as such
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test) # Convert to DataFrame if needed

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, predictions)
        print(f"R2 Score on test set: {r2}")

        if r2 > 0:
            print("Model prediction successful and R2 score is positive. Container verification passed.")
        else:
            print("Warning: R2 score is not positive or zero. Container verification might indicate an issue.")
            exit(1) # Exit with a non-zero code to indicate failure in CI

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure models are trained and saved. {e}")
        exit(1) # Exit with a non-zero code to indicate failure
    except Exception as e:
        print(f"An error occurred during prediction verification: {e}")
        exit(1) # Exit with a non-zero code to indicate failure

if __name__ == "__main__":
    verify_model_prediction()
