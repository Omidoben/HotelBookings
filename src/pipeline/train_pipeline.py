import os
import sys

from src.logger import logging
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import numpy as np

class TrainPipeline:
    """
    Orchestrates the complete machine learning training pipeline
    from data ingestion to model training.
    """
    def __init__(self):
        """Initializes the training pipeline."""
        logging.info("Training pipeline initialized")

    def run_pipeline(self):
        """
        Executes the complete training pipeline steps:
        1. Data Ingestion: Reads raw data, performs train-test split, saves artifacts.
        2. Data Transformation: Cleans, engineers features, preprocesses data, saves preprocessor.
        3. Model Training: Trains models, tunes hyperparameters, evaluates, selects best, saves model.

        Returns:
            float: The evaluation score (e.g., Test ROC AUC) of the best trained model.
                   Returns None or raises exception if pipeline fails.
        """
        logging.info("Starting the full training pipeline execution.")

        try:
            # Data Ingestion
            logging.info("--- Starting Data Ingestion step ---")
            data_ingestion = DataIngestion()
            # initiate_data_ingestion returns paths to train and test CSVs
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("--- Data Ingestion step completed ---")
            logging.info(f"Train data artifact saved at: {train_data_path}")
            logging.info(f"Test data artifact saved at: {test_data_path}")


            # Step 2: Data Transformation
            logging.info("--- Starting Data Transformation step ---")
            data_transformation = DataTransformation()
            # initiate_data_transformation takes paths, returns processed arrays and preprocessor path
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info("--- Data Transformation step completed ---")
            logging.info(f"Processed train array shape: {train_arr.shape}")
            logging.info(f"Processed test array shape: {test_arr.shape}")
            logging.info(f"Preprocessor object saved at: {preprocessor_path}")


            # Model Training
            logging.info("--- Starting Model Training step ---")
            model_trainer = ModelTrainer()
            # initiate_model_trainer takes processed arrays, returns best model's test score
            best_model_test_score = model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
            logging.info("--- Model Training step completed ---")
            logging.info(f"Best model's test evaluation score (ROC AUC): {best_model_test_score}")


            logging.info("Full training pipeline executed successfully.")

            # Return the final evaluation metric of the best model
            return best_model_test_score

        except CustomException as ce:
            logging.error(f"Custom exception occurred during pipeline execution: {ce}")
            print(f"Pipeline failed due to a custom exception: {ce}")
            raise 
        except Exception as e:
            # Catch any other unexpected exceptions
            logging.error(f"An unexpected error occurred during pipeline execution: {e}", exc_info=True)
            print(f"Pipeline failed due to an unexpected error: {e}")
            raise CustomException(e, sys)


# Main execution block
if __name__ == "__main__":
    # This block allows running the entire pipeline by executing this script directly
    logging.info("Executing the training pipeline from the main entry point.")
    try:
        # Create an instance of the TrainPipeline
        pipeline = TrainPipeline()

        # Run the pipeline
        final_score = pipeline.run_pipeline()

        logging.info(f"Pipeline finished. Final Best Model Test Score: {final_score}")
        print(f"\n--- Training Pipeline Completed ---")
        print(f"Final Best Model Test Score (ROC AUC): {final_score:.5f}")
        print(f"Check the 'logs' directory for detailed execution logs.")
        print(f"Check the 'artifacts' directory for saved data and model files.")


    except CustomException as ce:
        logging.error(f"Pipeline execution failed with a custom exception: {ce}")
        print(f"Pipeline execution failed: {ce}") # Print the formatted message
        sys.exit(1) # Exit with a non-zero status code to indicate failure
    except Exception as ex:
         # Catch any other unexpected exceptions not wrapped by CustomException
         logging.error(f"Pipeline execution failed with an unexpected error: {ex}", exc_info=True)
         print(f"Pipeline execution failed with an unexpected error: {ex}")
         sys.exit(1) # Exit with a non-zero status code to indicate failure

