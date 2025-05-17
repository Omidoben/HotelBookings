# Use an official Python runtime as a parent image
# Using a specific version like 3.10-slim ensures consistency and smaller image size
FROM python:3.10-slim

# Set the working directory in the container
# All subsequent commands will run from this directory
WORKDIR /app

# Copy the requirements.txt file into the container
# This step is done early to leverage Docker's build cache.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to disable cache, reducing image size
# Use --upgrade pip to ensure pip is up-to-date before installing
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This copies your src directory, app.py, etc.
COPY . .

# Optional: Set Python path if your imports require it (e.g., importing src modules)
# WORKDIR /app already makes /app the current directory, so relative imports from /app should work.
# If you have complex import structures, you might uncomment this:
# ENV PYTHONPATH="/app"

# Create the artifacts directory inside the container
# This directory will store your trained model and preprocessor files
# The training pipeline (if run during build) or a separate process will place files here
RUN mkdir -p artifacts

# --- Optional: Run the training pipeline during the Docker build ---
# Uncomment the line below if you want to train the model every time you build the image.
# This is useful for simple cases or if your data is part of the image,
# but often training is done separately or triggered on deployment/schedule.
# Make sure your train_pipeline.py can be run non-interactively.
# RUN python -m src.pipeline.train_pipeline

# Expose the port that the Flask application will listen on
# This doesn't actually publish the port, but documents which port is used
EXPOSE 8080

# Command to run the application using Gunicorn
# Gunicorn is a production-ready WSGI server
# --bind 0.0.0.0:8080 tells Gunicorn to listen on all network interfaces on port 8080
# app:app refers to the 'app' object inside the 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

# To build the Docker image:
# docker build -t hotel-booking-predictor .

# To run the Docker container (assuming artifacts/ are populated BEFORE building or by a build step):
# docker run -p 8080:8080 hotel-booking-predictor
