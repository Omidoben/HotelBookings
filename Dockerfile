FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create the artifacts directory inside the container
# This directory will store your trained model and preprocessor files
RUN mkdir -p artifacts

# Expose the port that the Flask application will listen on
EXPOSE 8080

# Command to run the application using Gunicorn
# --bind 0.0.0.0:8080 tells Gunicorn to listen on all network interfaces on port 8080
# app:app refers to the 'app' object inside the 'app.py' file
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]

# To build the Docker image:
# docker build -t hotel-booking-predictor .

# To run the Docker container (assuming artifacts/ are populated BEFORE building or by a build step):
# docker run -p 8080:8080 hotel-booking-predictor
