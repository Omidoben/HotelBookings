### Hotel Bookings Cancellation Project

Hotel Booking Cancellation PredictionAn end-to-end machine learning project to predict hotel booking cancellations. This project demonstrates a structured approach to building an ML pipeline, deploying a prediction service via a Flask application, and containerizing it with Docker. The application is deployed on Render.Project StructureThe project is organized into the following directories and files:.
├── artifacts/                  # Stores trained model and preprocessor (.pkl files)
├── logs/                       # Stores application logs
├── notebook/                   # Jupyter notebooks for initial exploration/development
│   └── data/                   # Raw data file (hotel_bookings.csv)
├── src/
│   ├── components/             # Individual ML pipeline components (data_ingestion, data_transformation, model_trainer)
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging configuration
│   ├── pipeline/               # Orchestration pipelines (train_pipeline, predict_pipeline)
│   └── utils.py                # Utility functions (saving/loading objects, model evaluation)
├── templates/                  # HTML templates for the Flask web application
│   ├── error.html
│   ├── form.html
│   ├── index.html
│   └── results.html
├── .dockerignore               # Specifies files/directories to ignore during Docker build
├── .gitignore                  # Specifies files/directories to ignore for Git
├── app.py                      # Flask web application entry point
├── Dockerfile                  # Instructions for building the Docker image
├── README.md                   # Project README file
├── requirements.txt            # Python dependencies
└── setup.py                    # Project packaging configuration
FeaturesData Ingestion: Reads raw data, handles duplicates, and performs stratified train-test splitting.Data Transformation: Includes feature engineering (e.g., total stay, guest count, cancellation rate, arrival day of week), handles missing values, scales numerical features, and encodes categorical features (One-Hot Encoding for low cardinality, Target Encoding for high cardinality).Model Training: Trains and evaluates multiple classification models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost) with hyperparameter tuning using cross-validation (ROC AUC scoring). Selects and saves the best performing model.Prediction Pipeline: Loads the saved preprocessor and model, applies the same transformation steps to new data, and makes predictions.Flask Application: Provides a web interface (form-based prediction) and a REST API endpoint (/api/predict) for receiving new booking data and returning cancellation predictions.Containerization: Uses Docker to package the application and its dependencies for consistent deployment.Deployment: Application is deployed on the Render platform.SetupClone the repository:git clone <repository_url>
cd <repository_name>
Create and activate a Conda environment:conda create -p venv python=3.10 -y
conda activate ./venv
If conda activate venv doesn't work, use the full path: conda activate E:\Projects\HotelCancellation\venv (replace with your actual path).If conda activate gives a CommandNotFoundError, run conda init <SHELL_NAME> (e.g., conda init powershell) and restart your terminal.Install dependencies:pip install -r requirements.txt
Running the Project1. Train the ModelRun the training pipeline to process the data, train models, and save the best model and preprocessor to the artifacts/ directory.python -m src.pipeline.train_pipeline
This will execute the DataIngestion, DataTransformation, and ModelTrainer components sequentially. Check the logs/ directory for detailed output.2. Run the Flask Application LocallyEnsure your Conda environment is active and the artifacts/ directory contains model.pkl and preprocessor.pkl.python app.py
The application will start, typically accessible at http://localhost:8080/.Access the web form at http://localhost:8080/predict.Send POST requests with JSON data to the API endpoint at http://localhost:8080/api/predict.3. Build and Run with DockerEnsure you have Docker installed and running.Build the Docker image:docker build -t orondoben/hotel-booking-predictor .
(Replace orondoben/hotel-booking-predictor with your desired image tag if different).Run the Docker container:docker run -d -p 8080:8080 --name hotel-booking-container orondoben/hotel-booking-predictor
This runs the container in detached mode (-d), maps host port 8080 to container port 8080 (-p 8080:8080), and names the container hotel-booking-container.The application will be accessible at http://localhost:8080/ via the Docker container.Deployment on RenderThe application has been deployed to the Render platform.Access the deployed application here:[Insert your Render App URL Here](Replace this placeholder with the actual URL provided by Render after deployment)The prediction service is available via the web interface and the /api/predict endpoint at the deployed URL.DependenciesSee the requirements.txt file for a list of all Python dependencies. Key libraries include:pandasnumpyscikit-learnxgboostcategory-encodersFlaskgunicorndillFuture ImprovementsImplement hyperparameter tuning for all models using a more exhaustive search or techniques like RandomizedSearchCV.Explore additional feature engineering ideas.Integrate data validation checks earlier in the pipeline.Set up CI/CD for automated testing and deployment.Implement model monitoring in production to detect data/concept drift.Add more robust error handling and logging, especially for edge cases in input data.Improve the web interface styling and user experience.Consider using a more scalable framework like FastAPI for the prediction service if high throughput is required.
