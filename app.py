from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import os
import sys

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging # Import logging for use in the app

# Initialize the Flask application
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    """Renders the home page."""
    logging.info("Rendering index.html")
    return render_template('index.html')

# Route for the prediction form and prediction results
@app.route('/predict', methods=['GET', 'POST'])
def predict_web():
    """
    Handles GET requests to display the prediction form and
    POST requests to make a prediction from form data.
    """
    logging.info(f"Received {request.method} request on /predict")
    try:
        if request.method == 'GET':
            # Render the form template for GET requests
            logging.info("Rendering form.html for GET request")
            return render_template('form.html')
        else: # Handle POST request
            logging.info("Processing POST request for prediction from form")

            # Get data from the form
            # Instantiate CustomData with data from the form
            try:
                data = CustomData(
                    hotel=request.form.get('hotel'),
                    is_canceled=int(request.form.get('is_canceled', 0)), # Default to 0 if not provided
                    lead_time=int(request.form.get('lead_time')),
                    arrival_date_year=int(request.form.get('arrival_date_year')),
                    arrival_date_month=request.form.get('arrival_date_month'),
                    arrival_date_week_number=int(request.form.get('arrival_date_week_number')),
                    arrival_date_day_of_month=int(request.form.get('arrival_date_day_of_month')),
                    stays_in_weekend_nights=int(request.form.get('stays_in_weekend_nights')),
                    stays_in_week_nights=int(request.form.get('stays_in_week_nights')),
                    adults=int(request.form.get('adults')),
                    children=int(request.form.get('children')),
                    babies=int(request.form.get('babies')),
                    meal=request.form.get('meal'),
                    country=request.form.get('country'),
                    market_segment=request.form.get('market_segment'),
                    distribution_channel=request.form.get('distribution_channel'),
                    is_repeated_guest=int(request.form.get('is_repeated_guest', 0)), # Default to 0
                    previous_cancellations=int(request.form.get('previous_cancellations', 0)), # Default to 0
                    previous_bookings_not_canceled=int(request.form.get('previous_bookings_not_canceled', 0)), # Default to 0
                    reserved_room_type=request.form.get('reserved_room_type'),
                    assigned_room_type=request.form.get('assigned_room_type'),
                    booking_changes=int(request.form.get('booking_changes', 0)), # Default to 0
                    deposit_type=request.form.get('deposit_type'),
                    agent=request.form.get('agent'), # Could be None or int, handle in CustomData if needed
                    company=request.form.get('company'), # Could be None or int, handle in CustomData if needed
                    days_in_waiting_list=int(request.form.get('days_in_waiting_list', 0)), # Default to 0
                    customer_type=request.form.get('customer_type'),
                    adr=float(request.form.get('adr')),
                    required_car_parking_spaces=int(request.form.get('required_car_parking_spaces', 0)), # Default to 0
                    total_of_special_requests=int(request.form.get('total_of_special_requests', 0)), # Default to 0
                    reservation_status=request.form.get('reservation_status'), # Will be dropped
                    reservation_status_date=request.form.get('reservation_status_date') # Will be dropped
                )
                logging.info("CustomData object created from form data.")

            except ValueError as ve:
                 logging.error(f"Value error in form data: {ve}")
                 # Render error template with specific message
                 return render_template('error.html', error=f"Invalid input data: {ve}. Please check numerical fields.")
            except Exception as data_e:
                 logging.error(f"Error processing form data: {data_e}")
                 # Render error template
                 return render_template('error.html', error=f"Error processing form data: {data_e}")

            # Convert CustomData to DataFrame
            input_df = data.get_data_as_dataframe()
            logging.info("Input data converted to DataFrame.")

            # Initialize and run prediction pipeline
            predict_pipeline = PredictPipeline()
            logging.info("PredictionPipeline initialized.")

            # Make prediction
            results = predict_pipeline.predict(input_df)
            predicted_cancellation = int(results[0]) # Get the single prediction as an integer

            logging.info(f"Prediction made: {predicted_cancellation}")

            # Render the results template
            return render_template('results.html', prediction=predicted_cancellation)

    except CustomException as ce:
        logging.error(f"Custom exception during web prediction: {ce}")
        return render_template('error.html', error=str(ce))
    except Exception as e:
        logging.error(f"An unexpected error occurred during web prediction: {e}", exc_info=True)
        return render_template('error.html', error=f"An unexpected error occurred: {e}")



# Route for making predictions via API (POST requests with JSON)
@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Handles POST requests to make a prediction from JSON data.
    Returns a JSON response.
    """
    logging.info("Received POST request on /api/predict")
    try:
        # --- Get data from JSON request body ---
        json_data = request.get_json() # Use get_json() to parse JSON body

        if not json_data:
             logging.warning("Received empty or non-JSON request body for API prediction.")
             return jsonify({
                 'status': 'error',
                 'message': 'Request body must be valid JSON.'
             }), 400 # Bad Request

        logging.info(f"Received JSON data: {json_data}")

        # --- Instantiate CustomData with data from JSON ---
        # Ensure you get ALL fields defined in your CustomData __init__
        # Handle potential missing fields or type conversion errors
        try:
            data = CustomData(
                hotel=json_data.get('hotel'),
                is_canceled=int(json_data.get('is_canceled', 0)), # Default to 0 if not provided
                lead_time=int(json_data.get('lead_time')),
                arrival_date_year=int(json_data.get('arrival_date_year')),
                arrival_date_month=json_data.get('arrival_date_month'),
                arrival_date_week_number=int(json_data.get('arrival_date_week_number')),
                arrival_date_day_of_month=int(json_data.get('arrival_date_day_of_month')),
                stays_in_weekend_nights=int(json_data.get('stays_in_weekend_nights')),
                stays_in_week_nights=int(json_data.get('stays_in_week_nights')),
                adults=int(json_data.get('adults')),
                children=int(json_data.get('children')),
                babies=int(json_data.get('babies')),
                meal=json_data.get('meal'),
                country=json_data.get('country'),
                market_segment=json_data.get('market_segment'),
                distribution_channel=json_data.get('distribution_channel'),
                is_repeated_guest=int(json_data.get('is_repeated_guest', 0)), # Default to 0
                previous_cancellations=int(json_data.get('previous_cancellations', 0)), # Default to 0
                previous_bookings_not_canceled=int(json_data.get('previous_bookings_not_canceled', 0)), # Default to 0
                reserved_room_type=json_data.get('reserved_room_type'),
                assigned_room_type=json_data.get('assigned_room_type'),
                booking_changes=int(json_data.get('booking_changes', 0)), # Default to 0
                deposit_type=json_data.get('deposit_type'),
                agent=json_data.get('agent'), # Could be None or int
                company=json_data.get('company'), # Could be None or int
                days_in_waiting_list=int(json_data.get('days_in_waiting_list', 0)), # Default to 0
                customer_type=json_data.get('customer_type'),
                adr=float(json_data.get('adr')),
                required_car_parking_spaces=int(json_data.get('required_car_parking_spaces', 0)), # Default to 0
                total_of_special_requests=int(json_data.get('total_of_special_requests', 0)), # Default to 0
                reservation_status=json_data.get('reservation_status'), # Will be dropped
                reservation_status_date=json_data.get('reservation_status_date') # Will be dropped
            )
            logging.info("CustomData object created from JSON data.")

        except ValueError as ve:
             logging.error(f"Value error in JSON data: {ve}")
             return jsonify({
                 'status': 'error',
                 'message': f"Invalid input data: {ve}. Please check numerical fields."
             }), 400 # Bad Request
        except Exception as data_e:
             logging.error(f"Error processing JSON data: {data_e}")
             return jsonify({
                 'status': 'error',
                 'message': f"Error processing JSON data: {data_e}"
             }), 400 # Bad Request


        # Convert CustomData to DataFrame
        input_df = data.get_data_as_dataframe()
        logging.info("Input data converted to DataFrame.")
        # logging.info(f"Input DataFrame:\n{input_df}") # Log the input DF for debugging


        # --- Initialize and run prediction pipeline ---
        # In a production app, load this once when the app starts, not on every request
        predict_pipeline = PredictPipeline()
        logging.info("PredictionPipeline initialized for API request.")

        # Make prediction
        # The predict method returns a numpy array, e.g., [0] or [1]
        results = predict_pipeline.predict(input_df)
        predicted_cancellation = int(results[0]) # Get the single prediction as an integer

        logging.info(f"API prediction made: {predicted_cancellation}")

        # Return JSON response
        return jsonify({
            'status': 'success',
            'prediction': predicted_cancellation # Return the integer prediction
        })

    except CustomException as ce:
        logging.error(f"Custom exception during API prediction: {ce}")
        # Return JSON error response
        return jsonify({
            'status': 'error',
            'message': str(ce) # Return the custom exception message
        }), 500 # Internal Server Error
    except Exception as e:
        logging.error(f"An unexpected error occurred during API prediction: {e}", exc_info=True)
        # Return JSON error response
        return jsonify({
            'status': 'error',
            'message': f"An unexpected error occurred: {e}"
        }), 500 # Internal Server Error


# --- Run the Flask app ---
if __name__ == "__main__":
    # Use debug=True for development. Set to False for production.
    # host='0.0.0.0' makes the server accessible externally (useful for Docker/deployment)
    app.run(host='0.0.0.0', port=8080, debug=True)
    # For production, use a WSGI server like Gunicorn (see later steps)
    # Example: gunicorn --workers 4 --bind 0.0.0.0:8080 app:app
