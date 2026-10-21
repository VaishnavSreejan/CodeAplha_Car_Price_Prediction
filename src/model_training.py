"""
Model Training and Evaluation Module for Car Price Prediction.
Trains multiple regression algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting),
evaluates them on held-out test data, extracts feature importances, and saves the best model.
"""

import os
import sys
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing import create_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "car_price_model.joblib")
DEFAULT_METRICS_PATH = os.path.join(OUTPUT_DIR, "model_metrics.json")
DEFAULT_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.csv")


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns candidate regression models.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=8, min_samples_split=5),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state, max_depth=10, min_samples_split=3),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=random_state)
    }


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Fit pipeline on training data and calculate MAE, MSE, RMSE, R² on held-out test data.
    """
    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    mae = float(mean_absolute_error(y_test, y_pred_test))
    mse = float(mean_squared_error(y_test, y_pred_test))
    rmse = float(np.sqrt(mse))
    r2_test = float(r2_score(y_test, y_pred_test))
    r2_train = float(r2_score(y_train, y_pred_train))

    return {
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2_Score": round(r2_test, 4),
        "Train_R2_Score": round(r2_train, 4),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], str, Pipeline]:
    """
    Trains all candidate models inside ColumnTransformer pipelines,
    evaluates them on the test set, and determines the best model based on R².
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    candidate_models = get_candidate_models(random_state)
    metrics_summary = {}
    fitted_pipelines = {}

    print("\n" + "=" * 65)
    print(f"{'MODEL EVALUATION BENCHMARK':^65}")
    print("=" * 65)

    for name, model in candidate_models.items():
        preprocessor = create_preprocessor()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        metrics = evaluate_model(pipe, X_train, y_train, X_test, y_test)
        metrics_summary[name] = metrics
        fitted_pipelines[name] = pipe

        print(f"[{name}]")
        print(f"  Test R2: {metrics['R2_Score']:.4f} | Train R2: {metrics['Train_R2_Score']:.4f}")
        print(f"  MAE: Rs. {metrics['MAE']:.3f}L | RMSE: Rs. {metrics['RMSE']:.3f}L | MSE: {metrics['MSE']:.3f}")
        print("-" * 65)

    # Select best model based on highest Test R2 Score
    best_model_name = max(metrics_summary, key=lambda k: metrics_summary[k]["R2_Score"])
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"\n[INFO] Best Performing Model: {best_model_name} (R2 = {metrics_summary[best_model_name]['R2_Score']:.4f})")

    # Save metrics to JSON
    with open(DEFAULT_METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[INFO] Saved metrics comparison to: {DEFAULT_METRICS_PATH}")

    # Save best pipeline
    joblib.dump(best_pipeline, DEFAULT_MODEL_PATH)
    print(f"[INFO] Saved trained model pipeline to: {DEFAULT_MODEL_PATH}")

    return metrics_summary, best_model_name, best_pipeline


def extract_feature_importance(
    best_pipeline: Pipeline,
    model_name: str,
    output_path: str = DEFAULT_IMPORTANCE_PATH
) -> pd.DataFrame:
    """
    Extracts feature names and their importance from the fitted pipeline.
    """
    preprocessor = best_pipeline.named_steps["preprocessor"]
    regressor = best_pipeline.named_steps["regressor"]

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    feature_names = NUMERICAL_FEATURES + encoded_cat_names

    if hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    elif hasattr(regressor, "coef_"):
        importances = np.abs(regressor.coef_)
    else:
        # Fallback equal importance
        importances = np.ones(len(feature_names)) / len(feature_names)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved feature importances to: {output_path}")

    return importance_df
