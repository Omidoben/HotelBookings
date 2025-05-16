import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any


from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from datetime import datetime


@dataclass
class PredictionPipelineConfig:
    """Configuration paths for Prediction Pipeline artifacts."""
    # Define paths where the trained model and preprocessor are saved
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class PredictPipeline:
    """
    Handles the prediction process: loading artifacts, transforming new data,
    and making predictions.
    """
    def __init__(self):
        """Initializes the prediction pipeline with configuration."""
        self.prediction_config = PredictionPipelineConfig()
        logging.info("Prediction pipeline initialized.")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Loads the trained model and preprocessor, transforms the input features,
        and makes predictions.

        Args:
            features (pd.DataFrame): Input features for prediction in a pandas DataFrame.
                                     This DataFrame should contain the raw features
                                     expected by the preprocessor *before* any
                                     transformation or feature engineering.

        Returns:
            np.ndarray: Predicted values (0 for not canceled, 1 for canceled).
        """
        logging.info("Starting prediction process.")
        try:
            # Load the Preprocessor and Model
            model_path = self.prediction_config.model_path
            preprocessor_path = self.prediction_config.preprocessor_path

            logging.info(f"Loading model from {model_path}")
            logging.info(f"Loading preprocessor from {preprocessor_path}")

            # Use the load_object utility to load the saved .pkl files
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            if model is None:
                 logging.error(f"Model not found at {model_path}")
                 raise FileNotFoundError(f"Model file not found at {model_path}")
            if preprocessor is None:
                 logging.error(f"Preprocessor not found at {preprocessor_path}")
                 raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

            logging.info("Model and preprocessor loaded successfully.")

            # Apply Feature Engineering and Transformations to Input Features 

            logging.info("Applying feature engineering and transformations to input features.")

            # Replicate the feature engineering steps from data_transformation.py
            def apply_feature_engineering_predict(df: pd.DataFrame) -> pd.DataFrame:
                df_copy = df.copy()

                # Ensure required columns exist before creating new ones
                required_cols_for_fe = [
                    "stays_in_weekend_nights", "stays_in_week_nights",
                    "adults", "children", "babies",
                    "previous_cancellations", "previous_bookings_not_canceled",
                    "arrival_date_year", "arrival_date_month", "arrival_date_day_of_month"
                ]
                if not all(col in df_copy.columns for col in required_cols_for_fe):
                     logging.warning("Missing one or more required columns for feature engineering.")

                # Create hotel_stay
                df_copy["hotel_stay"] = df_copy.get("stays_in_weekend_nights", 0) + df_copy.get("stays_in_week_nights", 0)

                # Create total_guests
                df_copy["total_guests"] = df_copy.get("adults", 0) + df_copy.get("children", 0) + df_copy.get("babies", 0)

                # Calculates previous cancellation rate, handle division by zero and fill NaNs
                prev_can = df_copy.get("previous_cancellations", 0)
                prev_not_can = df_copy.get("previous_bookings_not_canceled", 0)
                # Ensure calculation is safe even if source cols are missing or contain non-numeric
                with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for division by zero/NaN
                     df_copy["prev_cancellation_rate"] = (
                        prev_can / (prev_can + prev_not_can).replace(0, 1)
                    ).fillna(0) # Fill NaN/Inf resulting from division (if any) with 0


                # Mapping month names to numbers
                month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                if 'arrival_date_month' in df_copy.columns:
                    # Map month names, handle potential NaNs if month name is unexpected
                    df_copy['arrival_date_month_num'] = df_copy['arrival_date_month'].map(month_map).fillna(0)
                else:
                     df_copy['arrival_date_month_num'] = 0 # Create column with default 0 if source col missing


                # Create temporary datetime object for 'arrival_day_of_week' - handle errors
                date_cols_src = ['arrival_date_year', 'arrival_date_month_num', 'arrival_date_day_of_month']
                # Check if all source columns exist and are not entirely NaN/None before attempting conversion
                if all(col in df_copy.columns and df_copy[col].notna().any() for col in date_cols_src):
                    try:
                        # Create a temporary column 'temp_arrival_date'
                        df_copy["temp_arrival_date"] = pd.to_datetime(
                            df_copy["arrival_date_year"].astype(str) + '-' +
                            df_copy["arrival_date_month_num"].astype(str) + '-' + # Use numeric month
                            df_copy["arrival_date_day_of_month"].astype(str),
                            errors='coerce' # Will result in NaT for invalid dates
                        )
                        # Extract day of the week - will be NaN if temp_arrival_date is NaT
                        df_copy["arrival_day_of_week"] = df_copy["temp_arrival_date"].dt.day_name()
                        # Fill NaNs that result from NaT dates or missing source columns with 'Unknown'
                        df_copy['arrival_day_of_week'] = df_copy['arrival_day_of_week'].fillna('Unknown')

                        # Drop the temporary datetime column
                        df_copy.drop(columns=["temp_arrival_date"], inplace=True)
                    except Exception as date_e:
                         logging.warning(f"Error during arrival date processing: {date_e}. Setting arrival_day_of_week to 'Unknown'.")
                         df_copy["arrival_day_of_week"] = 'Unknown' # Default on error
                else:
                    logging.warning("Missing or entirely NaN source columns for arrival date. Setting arrival_day_of_week to 'Unknown'.")
                    df_copy["arrival_day_of_week"] = 'Unknown' # Create column with default if source cols missing


                return df_copy # Return DataFrame with new features


            # Apply feature engineering to the input features
            features_engineered = apply_feature_engineering_predict(features)
            logging.info(f"Features shape after engineering: {features_engineered.shape}")


            # Replicate the column dropping step from data_transformation.py
            cols_to_drop_after_fe: List[str] = [
                "arrival_date_month",    
                "arrival_date",         
                "company",               
                "agent",                 
                "reservation_status",    
                "reservation_status_date"
            ]
            # Drop columns that exist in the engineered features
            cols_to_drop_existing = [col for col in cols_to_drop_after_fe if col in features_engineered.columns]
            features_dropped = features_engineered.drop(columns=cols_to_drop_existing, errors='ignore') 
            logging.info(f"Dropped columns: {cols_to_drop_existing}")
            logging.info(f"Features shape after dropping: {features_dropped.shape}")


            # Replicate the Log1p transformation for 'adr' from data_transformation.py
            # Ensure 'adr' column exists and handle potential non-positive values before log1p
            if 'adr' in features_dropped.columns:
                # Replace non-positive values with a small epsilon or 0 before log1p
                # Using .clip(lower=0) to ensure values are not negative before log1p
                features_dropped['adr'] = np.log1p(features_dropped['adr'].clip(lower=0))

                logging.info("Applied log1p transformation to 'adr'. Relying on preprocessor for imputation.")
            else:
                 logging.warning("'adr' column not found after dropping. Skipping log1p transformation.")


            # Apply Preprocessing (Scaling, OHE, Target Encoding)
            logging.info("Applying preprocessor transformation to features.")

            # Transform the features using the loaded preprocessor
            data_scaled = preprocessor.transform(features_dropped)
            logging.info(f"Features shape after preprocessing: {data_scaled.shape}")


            # Make Predictions
            logging.info("Making predictions using the loaded model.")
            # The model expects the transformed features as a NumPy array
            predictions = model.predict(data_scaled)

            logging.info("Prediction process completed successfully.")
            return predictions # Returns 0 or 1 for classification


        except Exception as e:
            # Log the error with traceback before raising custom exception
            logging.error("Error during prediction process.", exc_info=True)
            raise CustomException(e, sys)


class CustomData:
    """
    Class to hold new input data for prediction and convert it to a DataFrame.
    Attributes correspond to the raw features expected by the prediction pipeline
    *before* any transformation or feature engineering.
    """
    def __init__(
        self,

        hotel: str, 
        is_canceled: int,
        lead_time: int,
        arrival_date_year: int,
        arrival_date_month: str,
        arrival_date_week_number: int, 
        arrival_date_day_of_month: int, 
        stays_in_weekend_nights: int, 
        stays_in_week_nights: int,
        adults: int, 
        children: int, 
        babies: int, 
        meal: str,
        country: str,
        market_segment: str, 
        distribution_channel: str,
        is_repeated_guest: int, 
        previous_cancellations: int, 
        previous_bookings_not_canceled: int, 
        reserved_room_type: str, 
        assigned_room_type: str, 
        booking_changes: int, 
        deposit_type: str, 
        agent: Any, 
        company: Any, 
        days_in_waiting_list: int, 
        customer_type: str, 
        adr: float, 
        required_car_parking_spaces: int, 
        total_of_special_requests: int, 
        reservation_status: str, 
        reservation_status_date: str 
        ):
        # Assign input values to instance attributes
        self.hotel = hotel
        self.is_canceled = is_canceled
        self.lead_time = lead_time
        self.arrival_date_year = arrival_date_year
        self.arrival_date_month = arrival_date_month
        self.arrival_date_week_number = arrival_date_week_number
        self.arrival_date_day_of_month = arrival_date_day_of_month
        self.stays_in_weekend_nights = stays_in_weekend_nights
        self.stays_in_week_nights = stays_in_week_nights
        self.adults = adults
        self.children = children
        self.babies = babies
        self.meal = meal
        self.country = country
        self.market_segment = market_segment
        self.distribution_channel = distribution_channel
        self.is_repeated_guest = is_repeated_guest
        self.previous_cancellations = previous_cancellations
        self.previous_bookings_not_canceled = previous_bookings_not_canceled
        self.reserved_room_type = reserved_room_type
        self.assigned_room_type = assigned_room_type
        self.booking_changes = booking_changes
        self.deposit_type = deposit_type
        self.agent = agent
        self.company = company
        self.days_in_waiting_list = days_in_waiting_list
        self.customer_type = customer_type
        self.adr = adr
        self.required_car_parking_spaces = required_car_parking_spaces
        self.total_of_special_requests = total_of_special_requests
        self.reservation_status = reservation_status
        self.reservation_status_date = reservation_status_date


    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Converts the instance attributes into a pandas DataFrame.

        Returns:
            pd.DataFrame: Input data as a DataFrame with columns
                          matching the expected raw data format.
        """
        try:
            # Create a dictionary from the instance attributes
            input_data_dict = {
                "hotel": [self.hotel],
                "is_canceled": [self.is_canceled], # Include if raw input might have it
                "lead_time": [self.lead_time],
                "arrival_date_year": [self.arrival_date_year],
                "arrival_date_month": [self.arrival_date_month],
                "arrival_date_week_number": [self.arrival_date_week_number],
                "arrival_date_day_of_month": [self.arrival_date_day_of_month],
                "stays_in_weekend_nights": [self.stays_in_weekend_nights],
                "stays_in_week_nights": [self.stays_in_week_nights],
                "adults": [self.adults],
                "children": [self.children],
                "babies": [self.babies],
                "meal": [self.meal],
                "country": [self.country],
                "market_segment": [self.market_segment],
                "distribution_channel": [self.distribution_channel],
                "is_repeated_guest": [self.is_repeated_guest],
                "previous_cancellations": [self.previous_cancellations],
                "previous_bookings_not_canceled": [self.previous_bookings_not_canceled],
                "reserved_room_type": [self.reserved_room_type],
                "assigned_room_type": [self.assigned_room_type],
                "booking_changes": [self.booking_changes],
                "deposit_type": [self.deposit_type],
                "agent": [self.agent],
                "company": [self.company],
                "days_in_waiting_list": [self.days_in_waiting_list],
                "customer_type": [self.customer_type],
                "adr": [self.adr],
                "required_car_parking_spaces": [self.required_car_parking_spaces],
                "total_of_special_requests": [self.total_of_special_requests],
                "reservation_status": [self.reservation_status],
                "reservation_status_date": [self.reservation_status_date]
            }
  
            return pd.DataFrame(input_data_dict)

        except Exception as e:
            logging.error("Error converting custom data to DataFrame.", exc_info=True)
            raise CustomException(e, sys)


# Example of how it can be used
if __name__ == "__main__":
    # This block demonstrates how to use the prediction pipeline with sample data.

    logging.info("\n--- Running Prediction Pipeline example ---")

    try:
        # Create sample input data using CustomData 
        sample_data = CustomData(
            hotel="Resort Hotel",
            is_canceled=0, # Provide a value, though it will be dropped internally
            lead_time=60,
            arrival_date_year=2016,
            arrival_date_month="April",
            arrival_date_week_number=15,
            arrival_date_day_of_month=20,
            stays_in_weekend_nights=2,
            stays_in_week_nights=3,
            adults=2,
            children=1, # With a child
            babies=0,
            meal="FB", # Full Board
            country="DEU", # Germany
            market_segment="Direct", # Direct booking
            distribution_channel="Direct",
            is_repeated_guest=0,
            previous_cancellations=0,
            previous_bookings_not_canceled=0,
            reserved_room_type="C",
            assigned_room_type="C",
            booking_changes=1, # Made a booking change
            deposit_type="No Deposit",
            agent=None,
            company=None,
            days_in_waiting_list=0,
            customer_type="Transient",
            adr=95.0,
            required_car_parking_spaces=1, # Requires parking
            total_of_special_requests=1, # Has a special request
            reservation_status="Check-Out",
            reservation_status_date="2016-04-25"
        )
        # Convert the sample data to a DataFrame using the CustomData method
        input_df = sample_data.get_data_as_dataframe()
        logging.info("Sample input data converted to DataFrame.")
    
        # Initialize the Prediction Pipeline
        predict_pipeline = PredictPipeline()
        logging.info("PredictionPipeline instance created.")


        # Make Prediction
        try:
            # Call the predict method with the input DataFrame
            prediction = predict_pipeline.predict(input_df)

            # The prediction is a NumPy array (e.g., [0] or [1])
            predicted_cancellation = prediction[0]

            logging.info(f"Prediction successful. Predicted value: {predicted_cancellation}")
            print(f"\n--- Prediction Result ---")
            print(f"Predicted Cancellation (0=No, 1=Yes): {predicted_cancellation}")


        except CustomException as ce:
            logging.error(f"A custom exception occurred during prediction: {ce}")
            print(f"Error during prediction: {ce}")
        except Exception as e:
             logging.error(f"An unexpected error occurred during prediction: {e}", exc_info=True)
             print(f"An unexpected error occurred during prediction: {e}")


    except CustomException as ce:
        # Catch exceptions from CustomData creation or other parts of this block
        logging.error(f"A custom exception occurred in __main__ block: {ce}")
        print(f"A custom exception occurred: {ce}")
    except Exception as ex:
         logging.error(f"An unexpected error occurred in __main__ block: {ex}", exc_info=True)
         print(f"An unexpected error occurred: {ex}")

