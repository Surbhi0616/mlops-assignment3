# quantize.py
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import os

# Define the PyTorch Linear Regression model
class PyTorchLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(PyTorchLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        return self.linear(x)

def manual_quantize_and_dequantize(param, scale, zero_point):
    # Quantize: (param / scale) + zero_point, then round and clip to 0-255
    quantized_param = np.round((param / scale) + zero_point).astype(np.uint8)
    # De-quantize: (quantized_param - zero_point) * scale
    dequantized_param = (quantized_param.astype(np.float32) - zero_point) * scale
    return quantized_param, dequantized_param

def run_quantization():
    print("Starting quantization process...")

    # 1. Load the saved scikit-learn model
    try:
        sklearn_model = joblib.load('models/linear_regression_model.joblib')
        print("Scikit-learn model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'models/linear_regression_model.joblib' not found. Please run train.py first.")
        exit(1)

    # Extract its learned parameters (coef_ and intercept_)
    sklearn_coef = sklearn_model.coef_
    sklearn_intercept = sklearn_model.intercept_
    print(f"Extracted Sklearn coef_ shape: {sklearn_coef.shape}, intercept_ shape: {sklearn_intercept.shape}")

    # 2. Store the unquantized parameters in a dictionary and save
    unquantized_params = {
        'coef': sklearn_coef,
        'intercept': sklearn_intercept
    }
    joblib.dump(unquantized_params, 'unquant_params.joblib')
    print(f"Unquantized parameters saved to unquant_params.joblib (Size: {os.path.getsize('unquant_params.joblib') / 1024:.2f} KB)")

    # 3. Perform manual quantization to an unsigned 8-bit integer
    # Determine min/max values for scaling
    all_params = np.concatenate([sklearn_coef.flatten(), sklearn_intercept.flatten()])
    q_min, q_max = 0, 255 # For unsigned 8-bit integer
    r_min, r_max = all_params.min(), all_params.max()

    # Calculate scale and zero-point
    scale = (r_max - r_min) / (q_max - q_min)
    zero_point = q_min - (r_min / scale)
    print(f"Quantization Scale: {scale}, Zero Point: {zero_point}")

    # Quantize and de-quantize coefficients
    quantized_coef, dequantized_coef = manual_quantize_and_dequantize(sklearn_coef, scale, zero_point)
    # Quantize and de-quantize intercept
    quantized_intercept, dequantized_intercept = manual_quantize_and_dequantize(sklearn_intercept, scale, zero_point)

    print(f"Quantized coef_ shape: {quantized_coef.shape}, intercept_ shape: {quantized_intercept.shape}")

    # 4. Store the quantized parameters in a dictionary and save
    quantized_params_dict = {
        'coef': quantized_coef,
        'intercept': quantized_intercept,
        'scale': scale,
        'zero_point': zero_point
    }
    joblib.dump(quantized_params_dict, 'quant_params.joblib')
    print(f"Quantized parameters saved to quant_params.joblib (Size: {os.path.getsize('quant_params.joblib') / 1024:.2f} KB)")

    # 5. Save the final, quantized PyTorch model (conceptually, we use dequantized weights)
    # Load test data to get input dimension for PyTorch model
    try:
        X_test, y_test = joblib.load('models/test_data.joblib')
    except FileNotFoundError:
        print("Error: 'models/test_data.joblib' not found. Please run train.py first.")
        exit(1)

    input_dim = X_test.shape[1]
    pytorch_model_quant = PyTorchLinearRegression(input_dim)

    # Manually set weights with dequantized parameters
    pytorch_model_quant.linear.weight.data = torch.tensor(dequantized_coef.reshape(1, -1), dtype=torch.float32)
    pytorch_model_quant.linear.bias.data = torch.tensor(dequantized_intercept, dtype=torch.float32)

    # Save the PyTorch model state_dict (or the entire model if preferred, but state_dict is common)
    torch.save(pytorch_model_quant.state_dict(), 'quantized_pytorch_model_state_dict.pth')
    print(f"Quantized PyTorch model state_dict saved to quantized_pytorch_model_state_dict.pth")


    # 6. Perform inference with the de-quantized weights and report R2 Score
    print("\nPerforming inference with de-quantized (quantized) model...")
    pytorch_model_quant.eval() # Set model to evaluation mode
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
        predictions_quant = pytorch_model_quant(X_test_tensor).numpy().flatten()
        r2_quant = r2_score(y_test, predictions_quant)
        print(f"R2 Score (Quantized Model): {r2_quant}")

    print("\nPerforming inference with original (unquantized) scikit-learn model for comparison...")
    predictions_orig = sklearn_model.predict(X_test)
    r2_orig = r2_score(y_test, predictions_orig)
    print(f"R2 Score (Original Sklearn Model): {r2_orig}")

    # Print results for report
    print("\n--- Summary for Report ---")
    print(f"Original Sklearn Model R2 Score: {r2_orig}")
    print(f"Quantized Model R2 Score: {r2_quant}")
    print(f"Original Model Params Size (unquant_params.joblib): {os.path.getsize('unquant_params.joblib') / 1024:.2f} KB")
    print(f"Quantized Model Params Size (quant_params.joblib): {os.path.getsize('quant_params.joblib') / 1024:.2f} KB")


if __name__ == "__main__":
    run_quantization()
