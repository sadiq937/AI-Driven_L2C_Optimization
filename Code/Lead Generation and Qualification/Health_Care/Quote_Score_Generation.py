import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from xgboost import XGBRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Quote Generation Score Calculation
def calculate_quote_generation_score(data):
    """
    Calculates quote generation score based on multiple pricing factors.

    :param data: Pandas DataFrame containing pricing model data
    :return: List of quote generation scores
    """
    weights = {
        "Treatment Cost Trends": 0.3,
        "Insurance Coverage": 0.25,
        "Hospital Pricing Models": 0.2,
        "Government Payer Adjustments": 0.25,
    }

    scores = []
    for _, row in data.iterrows():
        score = (
            row["Treatment Cost Trends"] * weights["Treatment Cost Trends"]
            + row["Insurance Coverage"] * weights["Insurance Coverage"]
            + row["Hospital Pricing Models"]
            * weights["Hospital Pricing Models"]
            + row["Government Payer Adjustments"]
            * weights["Government Payer Adjustments"]
        )
        scores.append(
            min(round(score * 100, 2), 100)
        )  # Normalize between 0 and 100
    return scores


# Function to add quote generation scores to dataset
def add_quote_scores_to_data(data):
    data["Quote Generation Score"] = calculate_quote_generation_score(data)
    return data


# Visualizations
def plot_pricing_distribution(data):
    """
    Plots the distribution of healthcare pricing trends.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data["Treatment Cost Trends"], bins=20, kde=True, color="blue"
    )
    plt.title("Distribution of Treatment Cost Trends")
    plt.xlabel("Treatment Cost Trend Score")
    plt.ylabel("Frequency")
    plt.show()


def plot_insurance_vs_quote(data):
    """
    Plots the relationship between insurance coverage and quote generation score.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=data["Insurance Coverage"],
        y=data["Quote Generation Score"],
        hue=data["Hospital Pricing Models"],
    )
    plt.title("Insurance Coverage vs. Quote Generation Score")
    plt.xlabel("Insurance Coverage Percentage")
    plt.ylabel("Quote Generation Score")
    plt.show()


# A/B Testing and Model Comparison
def build_and_evaluate_models(data):
    """
    Builds and evaluates predictive models: Linear Regression, Gradient Boosting, XGBoost, LightGBM, Neural Networks.
    """
    features = [
        "Treatment Cost Trends",
        "Insurance Coverage",
        "Hospital Pricing Models",
        "Government Payer Adjustments",
    ]
    target = "Quote Generation Score"

    X = data[features]
    y = data[target]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Linear Regression": LinearRegression(),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, random_state=42
        ),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, random_state=42),
    }

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"{model_name} Performance:")
        print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print("-" * 80)

    # Deep Learning Model: Neural Network
    logging.info("Training Neural Network Model...")

    nn_model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )
    nn_model.compile(loss="mse", optimizer="adam", metrics=["mae"])

    history = nn_model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    y_pred_nn = nn_model.predict(X_test).flatten()
    print(f"Neural Network Performance:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_nn):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred_nn):.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred_nn):.4f}")

    # Plot Training History
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Neural Network Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# Main function
def main():
    """
    Reads dataset, calculates quote scores, visualizes trends, and evaluates models.
    """
    try:
        # Load dataset
        data = pd.read_csv("quote_generation_data.csv")

        # Add quote generation scores
        data = add_quote_scores_to_data(data)

        # Save to Excel
        data.to_excel("updated_quote_generation_scores.xlsx", index=False)
        logging.info("Quote Generation Scores saved to Excel.")

        # Visualize data
        plot_pricing_distribution(data)
        plot_insurance_vs_quote(data)

        # Build and evaluate models
        build_and_evaluate_models(data)

    except Exception as e:
        logging.error(f"Error in processing: {e}")


if __name__ == "__main__":
    main()
