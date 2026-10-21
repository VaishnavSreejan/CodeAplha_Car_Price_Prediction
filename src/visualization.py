"""
Visualization Module for Car Price Prediction.
Generates exploratory data analysis (EDA) charts and model evaluation diagnostics.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Styling settings
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

DEFAULT_OUTPUT_DIR = os.path.join("outputs", "plots")


def ensure_output_dir(output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Ensure destination directory exists."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_price_distribution(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """1. Car price distribution histogram with KDE."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "price_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["Selling_Price"], kde=True, color="#2b5c8f", bins=25, ax=ax)
    ax.set_title("Distribution of Car Selling Prices (in Lakhs INR)", fontweight="bold", pad=12)
    ax.set_xlabel("Selling Price (₹ in Lakhs)")
    ax.set_ylabel("Frequency / Count")

    # Add mean and median lines
    mean_val = df["Selling_Price"].mean()
    median_val = df["Selling_Price"].median()
    ax.axvline(mean_val, color="#d9534f", linestyle="--", linewidth=1.8, label=f"Mean: ₹{mean_val:.2f}L")
    ax.axvline(median_val, color="#5cb85c", linestyle=":", linewidth=2, label=f"Median: ₹{median_val:.2f}L")
    ax.legend()

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_year_vs_price(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """2. Year versus Selling Price scatter plot with trend line."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "year_vs_price.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(
        data=df,
        x="Year",
        y="Selling_Price",
        scatter_kws={"alpha": 0.65, "color": "#1f77b4"},
        line_kws={"color": "#e74c3c", "linewidth": 2},
        ax=ax
    )
    ax.set_title("Vehicle Manufacturing Year vs. Selling Price", fontweight="bold", pad=12)
    ax.set_xlabel("Manufacturing Year")
    ax.set_ylabel("Selling Price (₹ in Lakhs)")
    ax.set_xticks(sorted(df["Year"].unique()))

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_km_vs_price(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """3. Kilometers driven versus Selling Price scatter plot."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "km_vs_price.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df,
        x="Driven_kms",
        y="Selling_Price",
        hue="Fuel_Type",
        style="Transmission",
        palette="tab10",
        s=70,
        alpha=0.75,
        ax=ax
    )
    ax.set_title("Kilometers Driven vs. Selling Price", fontweight="bold", pad=12)
    ax.set_xlabel("Kilometers Driven (kms)")
    ax.set_ylabel("Selling Price (₹ in Lakhs)")
    ax.legend(title="Vehicle Specs", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_price_by_category(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """4. Average price comparisons across Brand, Fuel Type, and Transmission."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "price_by_category.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fuel Type vs Price
    sns.barplot(
        data=df,
        x="Fuel_Type",
        y="Selling_Price",
        hue="Fuel_Type",
        legend=False,
        estimator=np.mean,
        errorbar=None,
        palette="Blues_d",
        ax=axes[0]
    )
    axes[0].set_title("Average Selling Price by Fuel Type", fontweight="bold")
    axes[0].set_xlabel("Fuel Type")
    axes[0].set_ylabel("Avg Selling Price (Lakhs INR)")
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(f"{height:.2f}L", (p.get_x() + p.get_width() / 2., height / 2),
                         ha='center', va='center', color='white', fontweight='bold')

    # Transmission vs Price
    sns.barplot(
        data=df,
        x="Transmission",
        y="Selling_Price",
        hue="Selling_type",
        estimator=np.mean,
        errorbar=None,
        palette="Set2",
        ax=axes[1]
    )
    axes[1].set_title("Average Selling Price by Transmission & Seller", fontweight="bold")
    axes[1].set_xlabel("Transmission")
    axes[1].set_ylabel("Avg Selling Price (Lakhs INR)")
    axes[1].legend(title="Seller Type")

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_price_by_brand(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Average selling price by brand."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "price_by_brand.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    brand_order = df.groupby("Brand")["Selling_Price"].mean().sort_values(ascending=False).index

    sns.barplot(
        data=df,
        x="Brand",
        y="Selling_Price",
        hue="Brand",
        legend=False,
        order=brand_order,
        estimator=np.mean,
        errorbar=None,
        palette="viridis",
        ax=ax
    )
    ax.set_title("Average Selling Price by Vehicle Brand", fontweight="bold", pad=12)
    ax.set_xlabel("Brand / Manufacturer")
    ax.set_ylabel("Average Selling Price (Lakhs INR)")
    plt.xticks(rotation=30, ha="right")

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"₹{height:.1f}L", (p.get_x() + p.get_width() / 2., height + 0.2),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """5. Correlation heatmap for numerical features."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[num_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Matrix of Numerical Features", fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_actual_vs_predicted(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
    model_name: str = "Best Model",
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """6. Actual versus Predicted Price scatter plot."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "actual_vs_predicted.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_actual, y_predicted, color="#3498db", alpha=0.75, edgecolors="k", s=60, label="Predictions")

    # Perfect prediction line (y = x)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#e74c3c", linestyle="--", linewidth=2, label="Ideal Fit (y = x)")

    ax.set_title(f"Actual vs. Predicted Selling Price ({model_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Actual Selling Price (₹ Lakhs)")
    ax.set_ylabel("Predicted Selling Price (₹ Lakhs)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_residuals(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
    model_name: str = "Best Model",
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """7. Residual plot to inspect error distribution."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "residual_plot.png")

    residuals = y_actual - y_predicted

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(y_predicted, residuals, color="#9b59b6", alpha=0.7, edgecolors="k", s=50)
    axes[0].axhline(0, color="#e74c3c", linestyle="--", linewidth=1.8)
    axes[0].set_title(f"Residuals vs. Fitted Values ({model_name})", fontweight="bold")
    axes[0].set_xlabel("Predicted Selling Price (₹ Lakhs)")
    axes[0].set_ylabel("Residuals (Actual - Predicted)")

    # Residual Distribution
    sns.histplot(residuals, kde=True, color="#2ecc71", bins=20, ax=axes[1])
    axes[1].axvline(0, color="#e74c3c", linestyle="--", linewidth=1.8)
    axes[1].set_title("Distribution of Residuals", fontweight="bold")
    axes[1].set_xlabel("Residual Value (₹ Lakhs)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Random Forest",
    top_n: int = 10,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    """8. Feature importance bar chart."""
    ensure_output_dir(output_dir)
    filepath = os.path.join(output_dir, "feature_importance.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    top_features = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

    bars = ax.barh(top_features["Feature"], top_features["Importance"], color="#34495e", edgecolor="#2c3e50")
    ax.set_title(f"Top {top_n} Influential Features ({model_name})", fontweight="bold", pad=12)
    ax.set_xlabel("Relative Feature Importance")
    ax.set_ylabel("Feature Name")

    # Annotate values
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f"{width:.3f}", (width + 0.005, bar.get_y() + bar.get_height() / 2),
                    ha="left", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    return filepath


def generate_all_eda_plots(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR) -> dict:
    """Generate all initial exploratory plots and return their file paths."""
    print("[INFO] Generating EDA visualizations...")
    plots = {
        "price_distribution": plot_price_distribution(df, output_dir),
        "year_vs_price": plot_year_vs_price(df, output_dir),
        "km_vs_price": plot_km_vs_price(df, output_dir),
        "price_by_category": plot_price_by_category(df, output_dir),
        "price_by_brand": plot_price_by_brand(df, output_dir),
        "correlation_heatmap": plot_correlation_heatmap(df, output_dir),
    }
    print(f"[INFO] Successfully generated {len(plots)} EDA plots in '{output_dir}'.")
    return plots
