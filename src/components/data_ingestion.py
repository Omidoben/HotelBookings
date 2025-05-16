import os
import sys

from src.exception import CustomException
from src.logger import logging
from typing import Tuple, List

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Configuration paths for Data Ingestion artifacts."""
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

# Creates a configuration class that holds default file paths for storing: Raw data (data.csv),
# Training data (train.csv), and Testing data (test.csv)


class DataIngestion:
    """Handles the data ingestion process."""

    def __init__(self):
        """Initializes with data ingestion configuration."""
        self.ingestion_config = DataIngestionConfig()
        logging.info("DataIngestion object initialized.")


    def initiate_data_ingestion(self) -> Tuple[str, str]:
        """
        Reads raw data, performs train-test split, and saves raw, train, and test data to artifacts.

        Returns:
            Tuple[str, str]: Paths to the saved training and testing data files.
        """
        logging.info("Entered the data ingestion method")

        try:
            # Defines the path to the raw data path
            raw_data_file_path = "notebook/data/hotel_bookings.csv" 

            # Ensure the raw data file exists
            if not os.path.exists(raw_data_file_path):
                 logging.error(f"Raw data file not found at {raw_data_file_path}")
                 raise FileNotFoundError(f"Raw data file not found at {raw_data_file_path}")


            # Read Raw Data
            df = pd.read_csv(raw_data_file_path)
            logging.info(f"Read the dataset as a dataframe from {raw_data_file_path}")
            logging.info(f"Initial dataset shape: {df.shape}")


            # Ensures that the artifacts/ directory exists; creates it if not
            artifacts_dir = os.path.dirname(self.ingestion_config.train_data_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            logging.info(f"Ensured artifacts directory exists: {artifacts_dir}")


            # Save Raw Data Artifact
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Saved raw data artifact to {self.ingestion_config.raw_data_path}")


            # Perform Train Test Split
            logging.info("Initiating train test split")

            # Define the target column name
            target_column_name = "is_canceled"

            # Ensure the target column exists in the DataFrame
            if target_column_name not in df.columns:
                 logging.error(f"Target column '{target_column_name}' not found in the dataset.")
                 raise ValueError(f"Target column '{target_column_name}' not found.")


            train_set, test_set = train_test_split(
                df,
                test_size=0.3,          
                random_state=42,        
                stratify=df[target_column_name] 
            )

            logging.info(f"Train test split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")


            # Save Train and Test Data Artifacts
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Saved train data artifact to {self.ingestion_config.train_data_path}")
            logging.info(f"Saved test data artifact to {self.ingestion_config.test_data_path}")


            logging.info("Data ingestion process completed successfully.")

            # Return the paths to the saved train and test data for the next pipeline stage - data transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error("Error during data ingestion process.", exc_info=True)
            raise CustomException(e, sys)


# Testing if above code works as expected
if __name__ == "__main__":
    logging.info("\n--- Running Data Ingestion step ---")
    try:
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()

        logging.info("Data Ingestion main execution completed successfully.")
        logging.info(f"Train data saved to: {train_data_path}")
        logging.info(f"Test data saved to: {test_data_path}")

        # Optional: Verify files were created
        # print(f"Train file exists: {os.path.exists(train_data_path)}")
        # print(f"Test file exists: {os.path.exists(test_data_path)}")


    except CustomException as ce:
        # The CustomException should already log the details inside its __init__ or the calling function
        logging.error(f"A custom exception occurred in __main__ block: {ce}")
        print(f"A custom exception occurred: {ce}") # Print the formatted message
    except Exception as ex:
         # Catch any other unexpected exceptions
         logging.error(f"An unexpected error occurred in __main__ block: {ex}", exc_info=True)
         print(f"An unexpected error occurred: {ex}")