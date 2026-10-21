"""
Prediction Module for Car Price Prediction.
Loads the trained Scikit-Learn pipeline and provides a robust, user-friendly
function for estimating used car prices.
"""

import os
import sys
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import CURRENT_YEAR, extract_brand, FEATURE_ORDER

DEFAULT_MODEL_PATH = os.path.join("models", "car_price_model.joblib")


def load_trained_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Load the saved pipeline from disk.
    """
    # Check current directory and parent directory for flexible loading
    search_paths = [
        model_path,
        os.path.join("..", model_path),
        os.path.abspath(model_path)
    ]
    for path in search_paths:
        if os.path.exists(path):
            return joblib.load(path)

    raise FileNotFoundError(
        f"Model file not found at '{model_path}'. "
        "Please run 'python main.py' to train and save the model pipeline first."
    )


def predict_car_price(
    features: Union[Dict[str, Any], pd.DataFrame],
    model_path: str = DEFAULT_MODEL_PATH
) -> Dict[str, Any]:
    """
    Predict the estimated selling price of a car given its features.

    Accepted dictionary keys:
    - Present_Price (float): Current ex-showroom price in Lakhs INR
    - Driven_kms (int): Total kilometers driven
    - Year (int) OR Car_Age (int): Manufacturing year or vehicle age
    - Fuel_Type (str): 'Petrol', 'Diesel', or 'CNG'
    - Selling_type (str): 'Dealer' or 'Individual'
    - Transmission (str): 'Manual' or 'Automatic'
    - Owner (int, optional): Number of previous owners (0, 1, 2, 3)
    - Brand (str, optional) OR Car_Name (str, optional): Vehicle brand or model name
    """
    model = load_trained_model(model_path)

    if isinstance(features, dict):
        data = features.copy()

        # Handle Car_Age / Year conversion
        if "Car_Age" not in data:
            if "Year" in data:
                data["Car_Age"] = max(CURRENT_YEAR - int(data["Year"]), 0)
            else:
                raise ValueError("Either 'Year' or 'Car_Age' must be provided.")

        # Handle Brand extraction if Car_Name is given
        if "Brand" not in data:
            if "Car_Name" in data:
                data["Brand"] = extract_brand(data["Car_Name"])
            else:
                data["Brand"] = "Other"

        # Default owner to 0 if not provided
        if "Owner" not in data:
            data["Owner"] = 0

        # Standardize strings
        data["Fuel_Type"] = str(data.get("Fuel_Type", "Petrol")).strip().capitalize()
        data["Selling_type"] = str(data.get("Selling_type", "Dealer")).strip().capitalize()
        data["Transmission"] = str(data.get("Transmission", "Manual")).strip().capitalize()

        df_input = pd.DataFrame([{col: data.get(col) for col in FEATURE_ORDER}])
    elif isinstance(features, pd.DataFrame):
        df_input = features.copy()
        if "Car_Age" not in df_input.columns and "Year" in df_input.columns:
            df_input["Car_Age"] = CURRENT_YEAR - df_input["Year"]
        if "Brand" not in df_input.columns and "Car_Name" in df_input.columns:
            df_input["Brand"] = df_input["Car_Name"].apply(extract_brand)
        df_input = df_input[FEATURE_ORDER]
    else:
        raise TypeError("Input 'features' must be a dictionary or a pandas DataFrame.")

    # Execute prediction
    prediction = model.predict(df_input)
    predicted_price = float(np.clip(prediction[0], 0.05, 100.0))
    price_in_rupees = int(round(predicted_price * 100000))

    return {
        "predicted_price_lakhs": round(predicted_price, 2),
        "predicted_price_inr": price_in_rupees,
        "formatted_display": f"Rs. {predicted_price:.2f} Lakhs (approx. Rs. {price_in_rupees:,})",
        "input_features": df_input.to_dict(orient="records")[0]
    }


if __name__ == "__main__":
    sample_car = {
        "Car_Name": "Honda City",
        "Year": 2017,
        "Present_Price": 9.85,
        "Driven_kms": 35000,
        "Fuel_Type": "Petrol",
        "Selling_type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0
    }
    try:
        result = predict_car_price(sample_car)
        print("Prediction Result:")
        print(result["formatted_display"])
    except Exception as e:
        print(f"Prediction error: {e}")
