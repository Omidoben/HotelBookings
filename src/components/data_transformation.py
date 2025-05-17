import os
import sys
from typing import List, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from category_encoders import TargetEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object 


@dataclass
class DataTransformationConfig:
    """Configuration paths for Data Transformation artifacts."""
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Handles data cleaning, feature engineering, and preprocessing steps."""

    def __init__(self):
        """Initializes with data transformation configuration."""
        self.data_transformation_config = DataTransformationConfig()
        logging.info("DataTransformation object initialized.")

        # Define the target column name for easy access
        self.target_column_name: str = "is_canceled"

        # Here, we define columns to drop based on feature engineering performed in jupyter notebook
        self.cols_to_drop_after_fe: List[str] = [
            "arrival_date_month",    
            "arrival_date",          
            "company",               
            "agent",                 
            "reservation_status",    
            "reservation_status_date"
        ]


    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Builds the preprocessing pipeline using ColumnTransformer.
        Defines transformations for numerical, low cardinality categorical (OHE),
        and high cardinality categorical (Target Encoding) columns.
        Assumes feature engineering and column dropping have already happened,
        so column names here must match the DataFrame state before this transformer is applied.
        """
        try:
            # Final Numerical Columns after engineering and dropping
            numerical_columns: List[str] = [
                'lead_time', 'arrival_date_year', 'arrival_date_week_number', 
                'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
                'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list', 
                'adr', 'required_car_parking_spaces', 'total_of_special_requests',
                'hotel_stay', 'total_guests', 'prev_cancellation_rate', 'arrival_date_month_num'
                
            ]

            # Final Low Cardinality Categorical Columns for One-Hot Encoding
            low_card_cat_columns: List[str] = [
                'hotel', 'meal', 'market_segment', 'distribution_channel',
                'deposit_type', 'customer_type', 'arrival_day_of_week'
            ]

            # Final High Cardinality Categorical Columns for Target Encoding
            high_card_cat_columns: List[str] = [
                'country', 'reserved_room_type', 'assigned_room_type'
            ]


            logging.info("Defining preprocessing steps:")
            logging.info(f"  Numerical columns: {numerical_columns}")
            logging.info(f"  Low Cardinality Categorical columns (OHE): {low_card_cat_columns}")
            logging.info(f"  High Cardinality Categorical columns (Target Encoding): {high_card_cat_columns}")


            # --- Create Pipelines ---

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")), 
                    ("scaler", StandardScaler())
                ]
            )

            # Low Cardinality Categorical Pipeline
            cat_low_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), 
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
                ]
            )

             # High Cardinality Categorical Pipeline:
            cat_high_pipeline = Pipeline(
                 steps=[
                     ("imputer", SimpleImputer(strategy="most_frequent")),
                     ("target_encoder", TargetEncoder()) 
                 ]
            )

            # --- Create Column Transformer ---
            # remainder='passthrough' keeps other columns not explicitly listed (e.g., if any were missed)
            preprocessor = ColumnTransformer(
                [
                    ("num", num_pipeline, numerical_columns),
                    ("cat_low", cat_low_pipeline, low_card_cat_columns),
                    ("cat_high", cat_high_pipeline, high_card_cat_columns)
                ],
                remainder='passthrough' 
            )

            logging.info("Preprocessing object (ColumnTransformer) successfully created with 3 pipelines.")

            return preprocessor

        except Exception as e:
            logging.error("Error building data transformer object.", exc_info=True)
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Reads data from paths, performs cleaning, feature engineering, applies preprocessing,
        and saves the fitted preprocessor object.

        Args:
            train_path (str): Path to the training data CSV (output from Data Ingestion).
            test_path (str): Path to the testing data CSV (output from Data Ingestion).

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: Transformed training array,
                                               Transformed testing array,
                                               Path to the saved preprocessor object.
        """
        try:
            # Read Data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed for transformation.")
            logging.info(f"Train data shape (initial): {train_df.shape}")
            logging.info(f"Test data shape (initial): {test_df.shape}")

            # Data Cleaning (Remove Duplicates & Reset Index)
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)
            train_df.reset_index(drop=True, inplace=True)
            test_df.reset_index(drop=True, inplace=True)

            logging.info("Dropped duplicates and reset index for train and test sets.")
            logging.info(f"Train data shape (after cleaning): {train_df.shape}")
            logging.info(f"Test data shape (after cleaning): {test_df.shape}")


            # Separate Features (X) and Target (y)
            input_feature_train_df = train_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_feature_test_df = test_df.drop(columns=[self.target_column_name], axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            logging.info(f"Separated target column '{self.target_column_name}' from features.")


            # Feature Engineering
            logging.info("Starting Feature Engineering...")

            # Define and apply the feature engineering 
            def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
                """Applies feature engineering steps based on notebook code."""
                df_copy = df.copy() 

                # Create hotel_stay
                df_copy["hotel_stay"] = df_copy["stays_in_weekend_nights"] + df_copy["stays_in_week_nights"]

                # Create total_guests
                df_copy["total_guests"] = df_copy["adults"] + df_copy["children"] + df_copy["babies"]

                # Calculates previous cancellation rate
                if 'previous_cancellations' in df_copy.columns and 'previous_bookings_not_canceled' in df_copy.columns:
                     df_copy["prev_cancellation_rate"] = (
                        df_copy["previous_cancellations"] /
                        (df_copy["previous_cancellations"] + df_copy["previous_bookings_not_canceled"]).replace(0, 1) # Replace 0 sum with 1 before division
                    ).fillna(0) # Fill NaN/Inf resulting from division (if any) with 0
                else:
                    df_copy["prev_cancellation_rate"] = 0 # Create column with default 0 if source cols missing


                # Mapping month names to numbers
                month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                if 'arrival_date_month' in df_copy.columns:
                    df_copy['arrival_date_month_num'] = df_copy['arrival_date_month'].map(month_map).fillna(0) # Fill potential NaNs if month name is unexpected
                else:
                     df_copy['arrival_date_month_num'] = 0 # Create column with default 0 if source col missing


                # Create temporary datetime object for 'arrival_day_of_week' - handle errors
                date_cols_src = ['arrival_date_year', 'arrival_date_month_num', 'arrival_date_day_of_month']
                if all(col in df_copy.columns for col in date_cols_src):
                    # Create a temporary column 'arrival_date'
                    df_copy["temp_arrival_date"] = pd.to_datetime(
                        df_copy["arrival_date_year"].astype(str) + '-' +
                        df_copy["arrival_date_month_num"].astype(str) + '-' +
                        df_copy["arrival_date_day_of_month"].astype(str),
                        errors='coerce' # Will result in NaT for invalid dates
                    )
                    # Extract day of the week - will be NaN if temp_arrival_date is NaT
                    df_copy["arrival_day_of_week"] = df_copy["temp_arrival_date"].dt.day_name()
                    # Fill NaNs that result from NaT dates or missing source columns with 'Unknown' or a common value
                    df_copy['arrival_day_of_week'].fillna('Unknown', inplace=True)

                    # Drop the temporary datetime column
                    df_copy.drop(columns=["temp_arrival_date"], inplace=True)
                else:
                    df_copy["arrival_day_of_week"] = 'Unknown' # Create column with default if source cols missing


                return df_copy # Return DataFrame with new features

            # Apply feature engineering to both train and test feature sets
            input_feature_train_df = apply_feature_engineering(input_feature_train_df)
            input_feature_test_df = apply_feature_engineering(input_feature_test_df)

            logging.info("Feature engineering completed.")
            logging.info(f"Train data shape (after FE): {input_feature_train_df.shape}")
            logging.info(f"Test data shape (after FE): {input_feature_test_df.shape}")


            # Drop Columns (after feature engineering, based on notebook)
            cols_to_drop_existing_train = [col for col in self.cols_to_drop_after_fe if col in input_feature_train_df.columns]
            cols_to_drop_existing_test = [col for col in self.cols_to_drop_after_fe if col in input_feature_test_df.columns]
            # Drop columns that exist in BOTH train and test feature sets
            cols_to_drop = list(set(cols_to_drop_existing_train) & set(cols_to_drop_existing_test))

            input_feature_train_df.drop(columns=cols_to_drop, inplace=True)
            input_feature_test_df.drop(columns=cols_to_drop, inplace=True)

            logging.info(f"Dropped columns: {cols_to_drop}")
            logging.info(f"Train data shape (after dropping): {input_feature_train_df.shape}")
            logging.info(f"Test data shape (after dropping): {input_feature_test_df.shape}")


            # Apply Numerical Transformations (Log1p for 'adr')
            # Apply log1p to 'adr' before standard scaling in the pipeline
            if 'adr' in input_feature_train_df.columns:
                input_feature_train_df['adr'] = np.log1p(input_feature_train_df['adr'])
                input_feature_test_df['adr'] = np.log1p(input_feature_test_df['adr'])

                # Fill potential NaNs created by log1p - Using median of the TRAIN set for imputation to avoid data leakage
                adr_median_train = input_feature_train_df['adr'].median() 
                input_feature_train_df['adr'] = input_feature_train_df['adr'].fillna(adr_median_train) 
                input_feature_test_df['adr'] = input_feature_test_df['adr'].fillna(adr_median_train)

                logging.info("Applied log1p transformation to 'adr' column and filled potential NaNs.")
            else:
                 logging.warning("'adr' column not found in features after FE and dropping. Skipping log1p transformation.")


            # Obtain and Apply Preprocessing Object (ColumnTransformer)
            logging.info("Obtaining preprocessing object...")
            preprocessing_obj = self.get_data_transformer_object() # Get the configured ColumnTransformer

            logging.info("Applying preprocessing object on training and testing data frames...")

            # Fit the transformer on the training data and transform both train and test data
            # Pass BOTH input features (X) and target (y) to fit_transform because TargetEncoder needs y
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df, target_feature_train_df
            )
            # Only pass X_test to transform
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Preprocessing (imputation, scaling, OHE, Target Encoding) applied successfully.")
            logging.info(f"Shape after preprocessing (train features): {input_feature_train_arr.shape}")
            logging.info(f"Shape after preprocessing (test features): {input_feature_test_arr.shape}")


            # Concatenate Features and Target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Concatenated transformed features and target for train and test sets.")
            logging.info(f"Final Train Array Shape: {train_arr.shape}")
            logging.info(f"Final Test Array Shape: {test_arr.shape}")


            # Save Preprocessor Object
            # Ensure the artifacts directory exists for the preprocessor file
            artifacts_dir = os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path)
            os.makedirs(artifacts_dir, exist_ok=True)
            logging.info(f"Ensured artifacts directory exists: {artifacts_dir}")

            # Save the FITTED preprocessor object using the util's function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return Results
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error("Error during data transformation process.", exc_info=True)
            raise CustomException(e, sys)


# For local testing
if __name__ == "__main__":
    # This block demonstrates how to run the transformation step.

    try:
        # Define paths to the data files created by data ingestion
        train_data_path_artifact = os.path.join("artifacts", "train.csv")
        test_data_path_artifact = os.path.join("artifacts", "test.csv")

        # Verify data files exist
        if not os.path.exists(train_data_path_artifact) or not os.path.exists(test_data_path_artifact):
             logging.error(f"Data files not found at {train_data_path_artifact} and {test_data_path_artifact}. Please ensure data ingestion was run successfully.")
             # Exit or raise error if data is missing
             sys.exit("Required data artifacts not found.")


        logging.info("\n--- Running Data Transformation step ---")
        # Create an instance of the DataTransformation class
        data_transformation = DataTransformation()

        # Run the data transformation process using the paths from data ingestion
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path_artifact,
            test_path=test_data_path_artifact
        )

        logging.info("Data Transformation main execution completed successfully.")
        logging.info(f"Final transformed Train Array Shape: {train_arr.shape}")
        logging.info(f"Final transformed Test Array Shape: {test_arr.shape}")
        logging.info(f"Preprocessor object saved at: {preprocessor_path}")

    except CustomException as ce:
        logging.error(f"A custom exception occurred in __main__ block: {ce}")
        print(f"A custom exception occurred: {ce}") 
    except Exception as ex:
         logging.error(f"An unexpected error occurred in __main__ block: {ex}", exc_info=True)
         print(f"An unexpected error occurred: {ex}")