import joblib
import os
import sys
import numpy as np
import pandas as pd
import time 

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer,
    classification_report
)


from src.exception import CustomException
from src.logger import logging

from typing import Dict, Any, Tuple



def save_object(file_path, obj):
    """
    Saves a Python object to a file using joblib.

    Args:
        file_path (str): The path to save the object.
        obj: The Python object to save.
    """
    try:
        # Ensure directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open in binary write mode
        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)
        logging.info(f"Object saved successfully to {file_path}")

    except Exception as e:
        logging.error(f"Error saving object to {file_path}: {e}", exc_info=True)
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from a file using joblib.

    Args:
        file_path (str): The path to the saved object file.

    Returns:
        obj: The loaded Python object.
    """
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return None

        # Open in binary read mode
        with open(file_path, "rb") as file_obj:
            obj = joblib.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}", exc_info=True)
        raise CustomException(e, sys)



def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                    models: Dict[str, Any], params: Dict[str, Dict[str, Any]], random_state: int = 42
                    ) -> Dict[str, Dict[str, Any]]:
    """
    Trains multiple classification models with hyperparameter tuning (GridSearchCV)
    and evaluates them on the test set.

    Args:
        X_train (np.ndarray): Training features (already preprocessed).
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Test features (already preprocessed).
        y_test (np.ndarray): Test target.
        models (Dict[str, Any]): Dictionary of model instances (e.g., {"LogisticRegression": LogisticRegression()}).
        params (Dict[str, Dict[str, Any]]): Dictionary of parameter grids for GridSearchCV.
        random_state (int): Random state for reproducibility.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing detailed results for each model:
                                   {'model_name': {'best_estimator': fitted_model,
                                                   'best_params': {...},
                                                   'best_cv_roc_auc': float, # Or other CV metric
                                                   'test_metrics': {...},
                                                   'train_time': float}}
    """
    report: Dict[str, Dict[str, Any]] = {}

    logging.info("Starting model evaluation and tuning process...")

    scoring_metric = 'roc_auc'

    for name, model in models.items():
        if name not in params:
             logging.warning(f"No parameter grid found for {name}. Skipping GridSearchCV and using default parameters.")
             param_grid = {}
        else:
            param_grid = params[name]


        start_time = time.time()
        logging.info(f"Training and tuning {name}...")

        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                scoring=scoring_metric, # Use ROC AUC as scoring metric for tuning
                verbose=1, 
                n_jobs=-1 
            )

            # Fit GridSearchCV on the training data
            grid_search.fit(X_train, y_train)

            # Get the best model found by GridSearchCV
            best_model = grid_search.best_estimator_

            # Evaluate the best model on the test set
            logging.info(f"Evaluating {name} (best estimator) on the test set...")

            # Make predictions
            y_pred = best_model.predict(X_test)

            # Calculate probability predictions for ROC AUC (if model supports it)
            y_pred_proba = None
            if hasattr(best_model, 'predict_proba'):
                # Get probabilities for the positive class (class 1)
                try:
                    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                except Exception as proba_e:
                     logging.warning(f"Could not get predict_proba for {name}: {proba_e}")
                     y_pred_proba = None


            # Calculate metrics
            test_roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            test_f1 = f1_score(y_test, y_pred, zero_division=0)

            # Get the full classification report for detailed view
            test_clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)


            # Store results in the report dictionary
            report[name] = {
                'best_estimator': best_model,
                'best_params': grid_search.best_params_,
                'best_cv_roc_auc': grid_search.best_score_, # Best score from GridSearchCV (CV ROC AUC)
                'test_metrics': {
                    'accuracy': test_accuracy,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1_score': test_f1,
                    'roc_auc': test_roc_auc,
                    'classification_report': test_clf_report # Store the full report
                },
                'train_time': time.time() - start_time
            }

            logging.info(f"{name} training and evaluation completed.")
            logging.info(f"  Best CV ROC AUC: {report[name]['best_cv_roc_auc']:.5f}")
            logging.info(f"  Test ROC AUC: {report[name]['test_metrics']['roc_auc']:.5f}" if not np.isnan(test_roc_auc) else f"  Test ROC AUC: N/A")
            logging.info(f"  Test F1 Score: {report[name]['test_metrics']['f1_score']:.5f}")
            logging.info(f"  Test Accuracy: {report[name]['test_metrics']['accuracy']:.5f}")
            logging.info(f"  Train Time: {report[name]['train_time']:.2f} seconds")
            logging.info(f"  Best Parameters: {report[name]['best_params']}")


        except Exception as e:
            logging.error(f"Error during training/evaluation of {name}: {e}", exc_info=True)
            report[name] = {'error': str(e)}


    logging.info("Model evaluation and tuning process completed.")
    return report