"""
Main Execution Script for CodeAlpha Task 3: Car Price Prediction with Machine Learning.
Orchestrates the entire end-to-end Data Science pipeline:
  1. Data loading and validation
  2. Data cleaning and feature engineering
  3. Exploratory Data Analysis (EDA) visualizations
  4. Train-test splitting without data leakage
  5. Multi-model training and comparison (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
  6. Model evaluation and selection
  7. Pipeline serialization (Joblib)
  8. Feature importance extraction
  9. Verification with sample predictions
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_loader import load_dataset, get_data_summary
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.visualization import (
    generate_all_eda_plots,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance
)
from src.model_training import (
    train_and_compare_models,
    extract_feature_importance,
    DEFAULT_MODEL_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_IMPORTANCE_PATH
)
from src.prediction import predict_car_price


def main():
    print("=" * 75)
    print("  CODEALPHA TASK 3: CAR PRICE PREDICTION WITH MACHINE LEARNING")
    print("=" * 75)

    # 1. Load Dataset
    print("\n>>> Step 1: Loading Dataset...")
    df_raw = load_dataset()
    summary = get_data_summary(df_raw)
    print(f"    - Dimensions: {summary['rows']} rows, {summary['columns']} columns")
    print(f"    - Missing Values: {summary['missing_values']}")
    print(f"    - Exact Duplicates: {summary['duplicates']}")

    # 2. Data Cleaning & Feature Engineering
    print("\n>>> Step 2: Cleaning & Engineering Features...")
    df_clean = clean_and_engineer_features(df_raw)
    print(f"    - Cleaned Dataset Dimensions: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"    - New Features Engineered: 'Car_Age', 'Brand'")
    print("    - Sample Engineered Records:")
    print(df_clean[["Car_Name", "Brand", "Year", "Car_Age", "Selling_Price", "Present_Price"]].head(3).to_string())

    # 3. Exploratory Data Analysis Plots
    print("\n>>> Step 3: Generating Exploratory Visualizations...")
    eda_plots = generate_all_eda_plots(df_clean)
    for plot_name, path in eda_plots.items():
        print(f"    - Generated: {path}")

    # 4. Train-Test Splitting (Prevent Data Leakage)
    print("\n>>> Step 4: Splitting Data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = prepare_train_test_data(df_raw, test_size=0.2, random_state=42)

    # 5. Train & Compare Multiple Regression Models
    print("\n>>> Step 5: Training and Benchmarking Regression Models...")
    metrics_summary, best_model_name, best_pipeline = train_and_compare_models(
        X_train, y_train, X_test, y_test, random_state=42
    )

    # 6. Diagnostics on Held-Out Test Data
    print(f"\n>>> Step 6: Generating Diagnostic Visualizations for Best Model ({best_model_name})...")
    y_test_pred = best_pipeline.predict(X_test)

    actual_pred_plot = plot_actual_vs_predicted(y_test, y_test_pred, model_name=best_model_name)
    residual_plot = plot_residuals(y_test, y_test_pred, model_name=best_model_name)
    print(f"    - Generated: {actual_pred_plot}")
    print(f"    - Generated: {residual_plot}")

    # 7. Feature Importance
    print("\n>>> Step 7: Extracting Feature Importance...")
    importance_df = extract_feature_importance(best_pipeline, best_model_name)
    feat_plot = plot_feature_importance(importance_df, model_name=best_model_name)
    print(f"    - Generated: {feat_plot}")
    print("    - Top 5 Influential Features:")
    print(importance_df.head(5).to_string(index=False))

    # 8. Model Verification with Test Predictions
    print("\n>>> Step 8: Verifying Saved Model with Sample Predictions...")
    sample_vehicles = [
        {
            "Car_Name": "Honda City",
            "Year": 2016,
            "Present_Price": 9.85,
            "Driven_kms": 32000,
            "Fuel_Type": "Petrol",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
            "Owner": 0
        },
        {
            "Car_Name": "Maruti Swift",
            "Year": 2014,
            "Present_Price": 6.87,
            "Driven_kms": 45000,
            "Fuel_Type": "Diesel",
            "Selling_type": "Dealer",
            "Transmission": "Manual",
            "Owner": 0
        },
        {
            "Car_Name": "Toyota Fortuner",
            "Year": 2015,
            "Present_Price": 30.61,
            "Driven_kms": 40000,
            "Fuel_Type": "Diesel",
            "Selling_type": "Dealer",
            "Transmission": "Automatic",
            "Owner": 0
        }
    ]

    for car in sample_vehicles:
        res = predict_car_price(car, model_path=DEFAULT_MODEL_PATH)
        print(f"    - Vehicle: {car['Car_Name']} ({car['Year']})")
        print(f"      Showroom Price: Rs. {car['Present_Price']}L | Kilometers: {car['Driven_kms']:,} km")
        print(f"      -> Predicted Selling Price: {res['formatted_display']}")

    # 9. Final Summary Table
    print("\n" + "=" * 75)
    print("  PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    print("\n[SUMMARY COMPARISON TABLE]")
    metrics_df = pd.DataFrame(metrics_summary).T[["R2_Score", "MAE", "RMSE", "MSE", "Train_R2_Score"]]
    metrics_df.columns = ["Test R2", "MAE (Rs. Lakhs)", "RMSE (Rs. Lakhs)", "MSE", "Train R2"]
    print(metrics_df.to_string())

    print("\n[KEY DELIVERABLE ARTIFACTS]")
    print(f"  * Trained Model Pipeline:  {DEFAULT_MODEL_PATH}")
    print(f"  * Performance Metrics:     {DEFAULT_METRICS_PATH}")
    print(f"  * Feature Importance CSV:  {DEFAULT_IMPORTANCE_PATH}")
    print(f"  * Generated Visualizations: outputs/plots/ (8 PNG figures)")
    print("\nYou can now launch the interactive web application with:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
