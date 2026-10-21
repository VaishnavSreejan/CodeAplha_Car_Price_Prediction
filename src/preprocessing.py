"""
Preprocessing and Feature Engineering Module for Car Price Prediction.
Handles data cleaning, brand extraction, age computation, outlier checks,
and building reproducible Scikit-Learn ColumnTransformers.
"""

from typing import Tuple, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CURRENT_YEAR = 2024

NUMERICAL_FEATURES = ["Present_Price", "Driven_kms", "Car_Age", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission", "Brand"]
FEATURE_ORDER = ["Present_Price", "Driven_kms", "Car_Age", "Fuel_Type", "Selling_type", "Transmission", "Owner", "Brand"]
TARGET_COLUMN = "Selling_Price"


def extract_brand(car_name: str) -> str:
    """
    Extract vehicle brand from Car_Name.
    Maps common model names to manufacturers (Maruti, Hyundai, Honda, Toyota, two-wheeler brands, etc.).
    """
    name = str(car_name).strip().lower()

    maruti_models = [
        "ritz", "sx4", "ciaz", "wagon r", "swift", "vitara brezza",
        "s cross", "alto 800", "ertiga", "dzire", "alto k10", "ignis",
        "800", "baleno", "omni"
    ]
    hyundai_models = [
        "i20", "grand i10", "i10", "eon", "xcent", "elantra", "creta",
        "verna", "santa fe"
    ]
    honda_models = [
        "city", "brio", "amaze", "jazz", "cr-v", "civic", "accord", "activa"
    ]
    toyota_models = [
        "corolla altis", "fortuner", "innova", "etios liva", "etios cross",
        "etios g", "corolla", "etios gd", "camry", "land cruiser"
    ]

    for model in maruti_models:
        if name.startswith(model):
            return "Maruti Suzuki"
    for model in hyundai_models:
        if name.startswith(model):
            return "Hyundai"
    for model in honda_models:
        if name.startswith(model) or "honda" in name:
            return "Honda"
    for model in toyota_models:
        if name.startswith(model):
            return "Toyota"

    if "royal enfield" in name:
        return "Royal Enfield"
    if "bajaj" in name:
        return "Bajaj"
    if "yamaha" in name:
        return "Yamaha"
    if "tvs" in name:
        return "TVS"
    if "hero" in name:
        return "Hero"
    if "ktm" in name:
        return "KTM"
    if "mahindra" in name:
        return "Mahindra"

    return "Other"


def clean_and_engineer_features(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Performs data cleaning, validation against invalid values,
    and engineers Car_Age and Brand features.
    """
    data = df.copy()

    # Check and remove exact duplicates if requested
    duplicate_count = data.duplicated().sum()
    if drop_duplicates and duplicate_count > 0:
        print(f"[INFO] Removing {duplicate_count} exact duplicate rows.")
        data = data.drop_duplicates().reset_index(drop=True)

    # Validate non-negative numbers
    invalid_rows = data[(data["Driven_kms"] < 0) | (data["Present_Price"] < 0) | (data["Selling_Price"] < 0)]
    if not invalid_rows.empty:
        print(f"[WARNING] Removing {len(invalid_rows)} rows with negative values.")
        data = data.drop(invalid_rows.index).reset_index(drop=True)

    # Feature Engineering 1: Compute Car Age
    data["Car_Age"] = CURRENT_YEAR - data["Year"]
    data["Car_Age"] = data["Car_Age"].apply(lambda age: max(age, 0))

    # Feature Engineering 2: Extract Brand
    data["Brand"] = data["Car_Name"].apply(extract_brand)

    # Standardize string columns
    for col in ["Fuel_Type", "Selling_type", "Transmission"]:
        data[col] = data[col].astype(str).str.strip().str.capitalize()

    return data


def create_preprocessor() -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer pipeline.
    Numerical features are standardized using StandardScaler.
    Categorical features are one-hot encoded with handle_unknown='ignore'.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and test sets without data leakage.
    """
    cleaned_df = clean_and_engineer_features(df)

    X = cleaned_df[FEATURE_ORDER]
    y = cleaned_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"[INFO] Train set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[INFO] Test set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    return X_train, X_test, y_train, y_test
