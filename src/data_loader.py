"""
Data Loader Module for Car Price Prediction.
Responsible for finding, loading, and performing initial validation on the dataset.
"""

import os
from typing import Optional
import pandas as pd

DEFAULT_LOCAL_PATHS = [
    os.path.join("data", "car_data.csv"),
    os.path.join("data", "car data.csv"),
    os.path.join("..", "data", "car_data.csv"),
    os.path.join("..", "data", "car data.csv"),
    r"C:\Users\vaish\Downloads\Car price\car data.csv"
]

EXPECTED_COLUMNS = [
    "Car_Name",
    "Year",
    "Selling_Price",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner"
]


def find_dataset_path(custom_path: Optional[str] = None) -> str:
    """Find the valid path to the dataset."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    for path in DEFAULT_LOCAL_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find car dataset. Please ensure 'car_data.csv' is placed in 'data/' folder."
    )


def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load dataset from CSV, validate required columns, and perform basic integrity checks.
    """
    valid_path = find_dataset_path(filepath)
    print(f"[INFO] Loading dataset from: {valid_path}")
    df = pd.read_csv(valid_path)

    # Validate columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    print(f"[INFO] Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary dictionary of dataset statistics.
    """
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numerical_columns": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist()
    }


if __name__ == "__main__":
    data = load_dataset()
    summary = get_data_summary(data)
    print("Summary:", summary)
    print(data.head())
