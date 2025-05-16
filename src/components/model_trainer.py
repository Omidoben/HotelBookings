# src/components/model_trainer.py

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any 
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb 

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    """Configuration paths for Model Trainer artifacts."""
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Handles training, evaluation, and saving of the best classification model."""

    def __init__(self):
        """Initializes with model trainer configuration."""
        self.model_trainer_config = ModelTrainerConfig()
        logging.info("ModelTrainer object initialized.")

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Initiates the model training and evaluation process.

        Args:
            train_array (np.ndarray): Processed training data (features and target concatenated).
            test_array (np.ndarray): Processed testing data (features and target concatenated).

        Returns:
            float: The ROC AUC score of the best model on the test set.
                   Returns the best CV ROC AUC if test evaluation fails or is skipped.
        """
        logging.info("Entered the model trainer method")

        try:
            logging.info("Splitting train and test arrays into features and target.")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


            # Define Models and Parameters for Tuning 
            models: Dict[str, Any] = {
                "LogisticRegression": LogisticRegression(random_state=42, solver='liblinear'),
                "RandomForestClassifier": RandomForestClassifier(random_state=42),
                "GradientBoostingClassifier": GradientBoostingClassifier(random_state=42),
                "XGBClassifier": xgb.XGBClassifier(
                    objective='binary:logistic', 
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42
                ),
            }

            params: Dict[str, Dict[str, Any]] = {
                "LogisticRegression": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ['l1', 'l2'] 
                },
                "RandomForestClassifier": {
                    "n_estimators": [100, 200, 300], 
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    # "min_samples_leaf": [1, 2],
                    # "max_features": ["sqrt", "log2"]
                },
                "GradientBoostingClassifier": {
                    "n_estimators": [100, 150, 200], 
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 4, 5], 
                    # "subsample": [0.8, 1.0],
                    # "criterion": ["friedman_mse", "squared_error"]
                },
                "XGBClassifier": {
                    'n_estimators': [100, 150, 200], 
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 4, 5], 
                    'colsample_bytree': [0.7, 0.8, 0.9] 
                    # 'subsample': [0.8, 0.9, 1.0],
                    # 'gamma': [0, 0.1, 0.2]
                },
            }

            logging.info(f"Defined {len(models)} classification models and {len(params)} parameter grids for tuning.")


            # Evaluate Models
            # Use the evaluate_models utility function to train and evaluate models
            logging.info("Starting model evaluation using evaluate_models utility...")
            model_report: Dict[str, Dict[str, Any]] = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
                random_state=42
            )

            logging.info("Model evaluation completed.")

            # Check if any models were evaluated successfully
            evaluated_models_report = {name: res for name, res in model_report.items() if 'error' not in res}

            if not evaluated_models_report:
                 raise CustomException("No models were evaluated successfully.", sys)


            # Select the Best Model
            # Find the best model based on the best cross-validation ROC AUC score obtained during tuning
            # Using max with a lambda function to find the key (model name) with the highest 'best_cv_roc_auc' value
            best_model_name = max(evaluated_models_report, key=lambda name: evaluated_models_report[name]['best_cv_roc_auc'])
            best_model_results = evaluated_models_report[best_model_name]
            best_model = best_model_results['best_estimator'] 
            best_model_cv_roc_auc = best_model_results['best_cv_roc_auc']
            best_model_test_roc_auc = best_model_results['test_metrics'].get('roc_auc', np.nan) # Get test ROC AUC if available
            best_model_test_f1 = best_model_results['test_metrics'].get('f1_score', np.nan) # Get test F1 if available


            logging.info(f"Best model selected based on CV ROC AUC: {best_model_name}")
            logging.info(f"  Best CV ROC AUC Score: {best_model_cv_roc_auc:.5f}")
            logging.info(f"  Test ROC AUC Score: {best_model_test_roc_auc:.5f}" if not np.isnan(best_model_test_roc_auc) else "  Test ROC AUC Score: N/A")
            logging.info(f"  Test F1 Score: {best_model_test_f1:.5f}" if not np.isnan(best_model_test_f1) else "  Test F1 Score: N/A")
            logging.info(f"  Best Parameters: {best_model_results['best_params']}")


            # Save the Best Model
            # Ensure the artifacts directory exists for the model file
            artifacts_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            logging.info(f"Ensured artifacts directory exists for model: {artifacts_dir}")

            # Save the best performing model object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model # Save the fitted best model
            )

            # Return the best roc auc metric
            return best_model_test_roc_auc


        except Exception as e:
            logging.error("Error during model training process.", exc_info=True)
            raise CustomException(e, sys)


# --- For local testing ---
if __name__ == "__main__":

    try:
        logging.info("\n--- Running Model Trainer step ---")

        logging.info("Simulating getting transformed arrays from Data Transformation...")
        try:
            from src.components.data_transformation import DataTransformation # Import Transformation component
            data_transformation = DataTransformation()
            # Assumes artifacts/train.csv and artifacts/test.csv exist from Data Ingestion
            train_data_path_artifact = os.path.join("artifacts", "train.csv")
            test_data_path_artifact = os.path.join("artifacts", "test.csv")

            if not os.path.exists(train_data_path_artifact) or not os.path.exists(test_data_path_artifact):
                 logging.error(f"Data files not found at {train_data_path_artifact} and {test_data_path_artifact}. Please ensure data ingestion was run successfully.")
                 sys.exit("Required data artifacts not found for Transformation.")

            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_path=train_data_path_artifact,
                test_path=test_data_path_artifact
            )
            logging.info("Successfully obtained transformed arrays by running Data Transformation.")

        except Exception as e:
             logging.error(f"Failed to obtain transformed arrays for testing: {e}", exc_info=True)
             sys.exit("Could not prepare data for Model Training test.")


        # Create an instance of the ModelTrainer class
        model_trainer = ModelTrainer()

        # Initiate the model training process
        best_model_test_score = model_trainer.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr
        )

        logging.info("Model Trainer main execution completed successfully.")
        logging.info(f"Best model's Test ROC AUC score: {best_model_test_score:.5f}")


    except CustomException as ce:
        logging.error(f"A custom exception occurred in __main__ block: {ce}")
        print(f"A custom exception occurred: {ce}")
    except Exception as ex:
         logging.error(f"An unexpected error occurred in __main__ block: {ex}", exc_info=True)
         print(f"An unexpected error occurred: {ex}")