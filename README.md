# 🏨 Hotel Booking Cancellation Prediction

An end-to-end machine learning project to predict hotel booking cancellations. This project demonstrates a structured approach to building an ML pipeline, deploying a prediction service via a Flask application, and containerizing it with Docker. The application is deployed on **Render**.

---

## 📁 Project Structure

```text
├── artifacts/                  # Stores trained model and preprocessor (.pkl files)
├── logs/                       # Stores application logs
├── notebook/                   # Jupyter notebooks for initial exploration/development
│   └── data/                   # Raw data file (hotel_bookings.csv)
├── src/
│   ├── components/             # ML pipeline components (data_ingestion, data_transformation, model_trainer)
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging configuration
│   ├── pipeline/               # Orchestration pipelines (train_pipeline, predict_pipeline)
│   └── utils.py                # Utility functions
├── templates/                  # HTML templates for the Flask web application
│   ├── error.html
│   ├── form.html
│   ├── index.html
│   └── results.html
├── .dockerignore               # Files/directories to ignore during Docker build
├── .gitignore                  # Files/directories to ignore for Git
├── app.py                      # Flask web application entry point
├── Dockerfile                  # Docker image build instructions
├── README.md                   # Project README file
├── requirements.txt            # Python dependencies
└── setup.py                    # Project packaging configuration
```

## Features

Data Ingestion: Reads raw data, handles duplicates, and performs stratified train-test splitting.

Data Transformation: Includes feature engineering (e.g., total stay, guest count, cancellation rate, arrival day of week), handles missing values, scales numerical features, and encodes categorical features (One-Hot Encoding for low cardinality, Target Encoding for high cardinality).

Model Training: Trains and evaluates multiple classification models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with hyperparameter tuning using cross-validation (ROC AUC scoring). Selects and saves the best performing model.

Prediction Pipeline: Loads the saved preprocessor and model, applies the same transformation steps to new data, and makes predictions.

Flask Application: Provides a web interface (form-based prediction) and a REST API endpoint (/api/predict) for receiving new booking data and returning cancellation predictions.

Containerization: Uses Docker to package the application and its dependencies for consistent deployment.

Deployment: Application is deployed on the Render platform.


### Setup
Clone the repository:

```
git clone <repository_url>
cd <repository_name>
```

Create and activate a Conda environment:
```
conda create -p venv python=3.10 -y
conda activate ./venv
```

If conda activate venv doesn't work, use the full path: conda activate 
```
E:\Projects\HotelCancellation\venv (replace with your actual path).
```

Install dependencies:
```
pip install -r requirements.txt
```

### Running the Project
**1. Train the Model**

Run the training pipeline to process the data, train models, and save the best model and preprocessor to the artifacts/ directory.
```
python -m src.pipeline.train_pipeline
```

This will execute the DataIngestion, DataTransformation, and ModelTrainer components sequentially. Check the logs/ directory for detailed output.

**2. Run the Flask Application Locally**

Ensure your Conda environment is active and the artifacts/ directory contains model.pkl and preprocessor.pkl.
```
python app.py
```

The application will start, typically accessible at http://localhost:8080/.

Access the web form at http://localhost:8080/predict.

Send POST requests with JSON data to the API endpoint at http://localhost:8080/api/predict.

**3. Build and Run with Docker**

Ensure you have Docker installed and running.

Build the Docker image:
```
docker build -t hotel-booking-predictor .
```
Run the Docker container:
```
docker run -d -p 8080:8080 --name hotel-booking-container hotel-booking-predictor
```
This runs the container in detached mode (-d), maps host port 8080 to container port 8080 (-p 8080:8080), and names the container hotel-booking-container.

The application will be accessible at http://localhost:8080/ via the Docker container.

**☁️ Deployment on Render**

The application has been deployed to the Render platform.

Access the deployed application here:

[https://hotelbookingspredictor.onrender.com]

The prediction service is available via the web interface and the /api/predict endpoint at the deployed URL.
