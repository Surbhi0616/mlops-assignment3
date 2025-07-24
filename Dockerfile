# Dockerfile
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code (train.py, predict.py, etc.)
# The 'models' directory will be created by train.py in the CI/CD pipeline
COPY . .

# Ensure the 'models' directory exists for consistency, though train.py will also create it
RUN mkdir -p models

# Default command to run your verification script (predict.py)
# This will be specifically overridden in the CI/CD workflow for testing
CMD ["python", "predict.py"]
