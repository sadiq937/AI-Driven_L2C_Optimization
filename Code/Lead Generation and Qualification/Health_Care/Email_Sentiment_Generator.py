import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Engagement Score Calculation
def calculate_engagement_score(data):
    """
    Calculates engagement scores based on various healthcare interactions.

    :param data: Pandas DataFrame containing engagement feature values
    :return: List of engagement scores
    """
    weights = {
        "Telehealth Engagement": 0.3,
        "Appointment Frequency": 0.3,
        "Support Requests": 0.2,
        "Patient Portal Usage": 0.2,
    }

    scores = []
    for _, row in data.iterrows():
        score = (
            row["Telehealth Engagement"] * weights["Telehealth Engagement"]
            + row["Appointment Frequency"] * weights["Appointment Frequency"]
            + row["Support Requests"] * weights["Support Requests"]
            + row["Patient Portal Usage"] * weights["Patient Portal Usage"]
        )
        scores.append(min(round(score, 2), 100))  # Normalize to 100
    return scores


# Function to add engagement scores to data
def add_engagement_scores_to_data(data):
    data["Engagement Score"] = calculate_engagement_score(data)
    return data


# Visualizations
def plot_feature_distributions(data):
    """
    Plots feature distributions using pairplots and violin plots.
    """
    sns.pairplot(
        data[
            [
                "Engagement Score",
                "Telehealth Engagement",
                "Appointment Frequency",
                "Support Requests",
                "Patient Portal Usage",
            ]
        ]
    )
    plt.suptitle("Feature Distributions", y=1.02)
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Patient Portal Usage",
        y="Engagement Score",
        data=data,
        palette="coolwarm",
    )
    plt.title("Engagement Score Distribution by Patient Portal Usage")
    plt.show()


def plot_correlation_matrix(data):
    """
    Plots correlation matrix for features.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True
    )
    plt.title("Feature Correlation Matrix")
    plt.show()


# A/B Testing and Model Comparison
def build_and_evaluate_models(data):
    """
    Builds and evaluates models using traditional, tree-based, and deep learning models.
    Uses A/B testing through cross-validation and hyperparameter tuning.
    """
    features = [
        "Telehealth Engagement",
        "Appointment Frequency",
        "Support Requests",
        "Patient Portal Usage",
    ]
    target = "Engagement Score"

    X = data[features]
    y = (data[target] >= 75).astype(
        int
    )  # Binary classification: High (1) vs Low (0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
        "Support Vector Machine (SVM)": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(50, 30), max_iter=500, random_state=42
        ),
    }

    for model_name, model in models.items():
        logging.info(f"Training {model_name}...")

        if model_name in ["Random Forest", "Gradient Boosting"]:
            param_grid = {"n_estimators": [50, 100, 200]}
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            feature_importances = model.feature_importances_
            sns.barplot(x=features, y=feature_importances)
            plt.title(f"Feature Importance - {model_name}")
            plt.show()

            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test)

        print("-" * 80)


# Main function
def main():
    """
    Reads data, calculates engagement scores, performs visualizations, and runs predictive models.
    """
    try:
        # Read dataset
        data = pd.read_csv("healthcare_engagement_data.csv")

        # Add engagement scores
        data = add_engagement_scores_to_data(data)

        # Save updated data to Excel
        data.to_excel("updated_healthcare_engagement_scores.xlsx", index=False)
        logging.info("Updated engagement scores saved to Excel.")

        # Visualize feature distributions
        plot_feature_distributions(data)

        # Visualize correlations
        plot_correlation_matrix(data)

        # Build and evaluate models
        build_and_evaluate_models(data)

    except Exception as e:
        logging.error(f"Error in processing: {e}")


if __name__ == "__main__":
    main()
