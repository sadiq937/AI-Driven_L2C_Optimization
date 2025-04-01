import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Helper function to map categorical data to numerical values for scoring
def map_specialty(specialty):
    specialty_map = {
        "Cardiology": 80,
        "Neurology": 70,
        "Orthopedics": 60,
        "General Surgery": 50,
        "Pediatrics": 40,
    }
    return specialty_map.get(specialty, 0)


def map_engagement_history(engagement):
    return (
        int(engagement.split()[1]) * 2
    )  # e.g., "Visited 4 times" -> 4 * 2 = 8 score


def map_referral_source(referral_source):
    referral_map = {
        "Doctor Referral": 80,
        "Digital Ad": 70,
        "Word of Mouth": 60,
        "Online Search": 50,
        "Direct Visit": 40,
    }
    return referral_map.get(referral_source, 0)


def map_lead_source(lead_source):
    lead_map = {"Digital Ad": 70, "Referral": 80, "Organic Search": 90}
    return lead_map.get(lead_source, 0)


# Function to calculate Lead Score
def calculate_lead_score(data):
    weights = {
        "Specialty": 20,
        "Engagement History": 15,
        "Referral Source": 10,
        "Website Interactions": 15,
        "Lead Source": 10,
        "Conversion Probability": 30,
    }

    lead_scores = []
    for index, row in data.iterrows():
        specialty_score = map_specialty(row["Specialty"])
        engagement_score = map_engagement_history(row["Engagement History"])
        referral_source_score = map_referral_source(row["Referral Source"])
        website_interaction_score = row["Website Interactions"] * 2
        lead_source_score = map_lead_source(row["Lead Source"])
        conversion_score = row["Conversion Probability"] * 0.3

        total_score = (
            (specialty_score * weights["Specialty"])
            + (engagement_score * weights["Engagement History"])
            + (referral_source_score * weights["Referral Source"])
            + (website_interaction_score * weights["Website Interactions"])
            + (lead_source_score * weights["Lead Source"])
            + (conversion_score * weights["Conversion Probability"])
        ) / sum(weights.values())

        lead_scores.append(round(total_score, 2))

    return lead_scores


# Function to add lead scores to the dataset
def add_lead_scores_to_data(data):
    data["Lead Score"] = calculate_lead_score(data)
    return data


# Advanced Visualizations


def plot_feature_distributions(data):
    """
    Create pair plots and violin plots for feature distributions to understand relationships
    """
    sns.pairplot(
        data[
            [
                "Lead Score",
                "Engagement History",
                "Website Interactions",
                "Referral Source",
                "Conversion Probability",
            ]
        ]
    )
    plt.suptitle("Feature Distributions and Pairwise Relationships", y=1.02)
    plt.show()

    # Violin Plot to observe distribution of Lead Score for different Referral Sources
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Referral Source", y="Lead Score", data=data, palette="Set2"
    )
    plt.title("Lead Score Distribution by Referral Source")
    plt.show()


def plot_correlation_matrix(data):
    corr_data = data[
        [
            "Lead Score",
            "Engagement History",
            "Website Interactions",
            "Referral Source",
            "Conversion Probability",
        ]
    ].copy()
    corr_data["Engagement History"] = corr_data["Engagement History"].apply(
        lambda x: int(x.split()[1])
    )

    correlation_matrix = corr_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True
    )
    plt.title("Correlation Matrix for Lead Score Prediction")
    plt.show()


# A/B Testing and Model Comparison


def build_and_evaluate_models(data):
    """
    Build and compare multiple models including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
    A/B testing approach by cross-validation and hyperparameter tuning.
    """
    # Features and target variable
    data["Referral Source"] = data["Referral Source"].apply(
        map_referral_source
    )
    data["Lead Source"] = data["Lead Source"].apply(map_lead_source)
    data["Engagement History"] = data["Engagement History"].apply(
        lambda x: int(x.split()[1])
    )

    features = [
        "Engagement History",
        "Website Interactions",
        "Referral Source",
        "Lead Source",
        "Conversion Probability",
    ]
    target = "Lead Score"

    X = data[features]
    y = (data[target] >= 75).astype(
        int
    )  # Binary outcome: High Lead Score = 1, Low Lead Score = 0

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    }

    # Cross-validation and GridSearchCV for hyperparameter tuning
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}...")

        # Hyperparameter tuning for Random Forest and Gradient Boosting
        if model_name == "Random Forest":
            param_grid = {
                "max_depth": [5, 10, None],
                "n_estimators": [50, 100, 200],
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        elif model_name == "Gradient Boosting":
            param_grid = {
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [100, 200, 300],
            }
            grid_search = GridSearchCV(
                model, param_grid, cv=3, n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate performance on test set
        y_pred = model.predict(X_test)
        print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        # Plot Feature Importance
        if model_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
            feature_importances = model.feature_importances_
            sns.barplot(x=features, y=feature_importances)
            plt.title(f"Feature Importance for {model_name}")
            plt.show()

            # SHAP values for interpretation
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test)

        print("\n" + "-" * 80 + "\n")


# Main function to process the data, calculate lead scores, visualize, and perform model comparison
def main():
    """
    Main function to read the data from CSV, calculate lead scores, visualize the data,
    and perform predictive analysis.
    """
    # Read the existing dataset from the CSV file
    data = pd.read_csv("healthcare_lead_scores.csv")

    # Add the lead scores to the data
    data_with_scores = add_lead_scores_to_data(data)

    # Visualize the Feature Distributions
    plot_feature_distributions(data_with_scores)

    # Visualize the Correlation Matrix
    plot_correlation_matrix(data_with_scores)

    # Build and evaluate multiple models
    build_and_evaluate_models(data_with_scores)

    # Optionally, save the updated data to a new CSV file
    data_with_scores.to_csv(
        "updated_healthcare_lead_scores_with_predictions.csv", index=False
    )
    print(
        "Lead scores and predictive analysis have been saved to updated_healthcare_lead_scores_with_predictions.csv"
    )


# Execute the main function
if __name__ == "__main__":
    main()
