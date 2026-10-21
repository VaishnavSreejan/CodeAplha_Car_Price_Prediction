"""
Interactive Streamlit Web Application for Car Price Prediction.
CodeAlpha Task 3: Machine Learning Project.
Run via:
  streamlit run app.py
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(
    page_title="Car Price Predictor | CodeAlpha Task 3",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.prediction import predict_car_price, load_trained_model, DEFAULT_MODEL_PATH
from src.preprocessing import CURRENT_YEAR, extract_brand

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-price {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #6B7280;
        border-left: 3px solid #F59E0B;
        padding-left: 10px;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def check_model_exists():
    """Verify that the model file exists on disk."""
    return os.path.exists(DEFAULT_MODEL_PATH)


def load_metrics_and_importance():
    """Load evaluation metrics and feature importances if available."""
    metrics_path = os.path.join(PROJECT_ROOT, "outputs", "model_metrics.json")
    importance_path = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.csv")

    metrics = None
    importance = None

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    if os.path.exists(importance_path):
        importance = pd.read_csv(importance_path)

    return metrics, importance


def main():
    # Header Banner
    st.markdown('<div class="main-header">🚗 Car Price Prediction System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">CodeAlpha Data Science Virtual Internship (Task 3) • '
        'Trained Regression Model with Scikit-Learn Pipeline</div>',
        unsafe_allow_html=True
    )

    if not check_model_exists():
        st.error(
            "⚠️ Trained model pipeline not found at `models/car_price_model.joblib`. "
            "Please run `python main.py` in your terminal to train and save the model before using this application."
        )
        return

    # Sidebar Information
    with st.sidebar:
        st.header("📌 Project Details")
        st.markdown("""
        **Internship Provider:** CodeAlpha  
        **Task:** 3 — Car Price Prediction  
        **Domain:** Machine Learning & Data Science  
        **Author:** Data Science Intern  
        """)

        metrics, importance = load_metrics_and_importance()
        if metrics:
            st.markdown("---")
            st.subheader("📊 Model Performance")
            for model_name, m in metrics.items():
                st.markdown(f"**{model_name}**")
                st.caption(f"Test R²: `{m['R2_Score']:.4f}` | MAE: `₹{m['MAE']:.2f}L` | RMSE: `₹{m['RMSE']:.2f}L`")

        st.markdown("---")
        st.caption("ℹ️ Currency Scale: 1.00 Lakh = ₹1,00,000 INR")

    # Main Form Layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🔧 Enter Vehicle Specifications")

        popular_cars = [
            "Honda City", "Maruti Swift", "Hyundai i20", "Toyota Fortuner",
            "Toyota Innova", "Maruti Ciaz", "Hyundai Verna", "Maruti Dzire",
            "Maruti Wagon R", "Honda Brio", "Royal Enfield Classic 350", "Custom / Other"
        ]
        car_choice = st.selectbox("Select Vehicle Model (or type custom)", popular_cars, index=0)
        if car_choice == "Custom / Other":
            car_name_input = st.text_input("Enter Car / Brand Name", value="Honda City")
        else:
            car_name_input = car_choice

        year_val = st.slider(
            "Manufacturing Year",
            min_value=2000,
            max_value=CURRENT_YEAR,
            value=2016,
            step=1,
            help="Year the vehicle was registered / manufactured."
        )
        calculated_age = max(CURRENT_YEAR - year_val, 0)
        st.caption(f"Calculated Vehicle Age: **{calculated_age} years**")

        present_price = st.number_input(
            "Original Showroom Price (in ₹ Lakhs)",
            min_value=0.20,
            max_value=100.0,
            value=9.85,
            step=0.25,
            help="Showroom ex-showroom price when purchased new (1 Lakh = ₹100,000)."
        )

        driven_kms = st.number_input(
            "Total Kilometers Driven",
            min_value=500,
            max_value=500000,
            value=35000,
            step=1000,
            help="Odometer reading in kilometers."
        )

    with col2:
        st.subheader("⚙️ Technical & Ownership Details")

        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"], index=0)
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"], index=0)
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"], index=0)
        owners = st.selectbox("Number of Previous Owners", [0, 1, 2, 3], index=0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Estimated Selling Price", type="primary", use_container_width=True)

        if predict_btn:
            vehicle_payload = {
                "Car_Name": car_name_input,
                "Year": int(year_val),
                "Present_Price": float(present_price),
                "Driven_kms": int(driven_kms),
                "Fuel_Type": fuel_type,
                "Selling_type": seller_type,
                "Transmission": transmission,
                "Owner": int(owners)
            }

            try:
                with st.spinner("Analyzing market valuation..."):
                    result = predict_car_price(vehicle_payload, DEFAULT_MODEL_PATH)

                pred_lakhs = result["predicted_price_lakhs"]
                pred_inr = result["predicted_price_inr"]
                depreciation = max(present_price - pred_lakhs, 0.0)
                retained_val = min(max((pred_lakhs / present_price) * 100, 0.0), 100.0)

                st.markdown(f"""
                <div class="result-card">
                    <div>ESTIMATED RESALE VALUATION</div>
                    <div class="result-price">₹ {pred_lakhs:.2f} Lakhs</div>
                    <div>Approximately <b>₹ {pred_inr:,} INR</b></div>
                </div>
                """, unsafe_allow_html=True)

                # Valuation Metrics
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Est. Value Retained", f"{retained_val:.1f}%")
                m_col2.metric("Total Depreciation", f"₹ {depreciation:.2f} L")
                m_col3.metric("Vehicle Age", f"{calculated_age} yrs")

            except Exception as e:
                st.error(f"Prediction failed with error: {str(e)}")

        st.markdown("""
        <div class="disclaimer-box">
            <b>Disclaimer:</b> Predictions are generated using machine learning models trained on historical
            market transactions. Actual market prices may fluctuate depending on cosmetic wear, vehicle condition,
            location, insurance, and negotiation.
        </div>
        """, unsafe_allow_html=True)

    # Detailed Analysis Tabs
    st.markdown("---")
    tab1, tab2 = st.tabs(["📈 Model Visualizations & Metrics", "📋 Feature Importance Insights"])

    with tab1:
        st.subheader("Model Evaluation Diagnostics")
        pcol1, pcol2 = st.columns(2)
        plot_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")

        actual_pred_img = os.path.join(plot_dir, "actual_vs_predicted.png")
        residual_img = os.path.join(plot_dir, "residual_plot.png")

        if os.path.exists(actual_pred_img):
            pcol1.image(actual_pred_img, caption="Actual vs. Predicted Prices on Test Set", use_container_width=True)
        if os.path.exists(residual_img):
            pcol2.image(residual_img, caption="Residual & Error Distribution", use_container_width=True)

    with tab2:
        st.subheader("Top Predictive Factors")
        if importance is not None:
            st.dataframe(importance.head(10), use_container_width=True)
            feat_plot = os.path.join(plot_dir, "feature_importance.png")
            if os.path.exists(feat_plot):
                st.image(feat_plot, caption="Feature Importance Ranking", use_container_width=True)


if __name__ == "__main__":
    main()
