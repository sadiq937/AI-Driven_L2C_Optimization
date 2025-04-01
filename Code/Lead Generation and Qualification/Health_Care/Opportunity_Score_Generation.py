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
from xgboost import XGBClassifier

# Helper function to process and map categorical variables to numerical values


def map_demographics(demographics):
    """
    Maps demographics to numerical values.
    Example: Age group and gender mapped to values.
    """
    if demographics == "Male":
        return 1
    elif demographics == "Female":
        return 2
    elif demographics == "California":
        return 10  # Assume California has higher potential
    elif demographics == "Texas":
        return 8
    return 5  # Default for unknown demographics


def map_healthcare_needs(healthcare_needs):
    """
    Maps healthcare needs (e.g., services required) to numerical scores.
    """
    healthcare_map = {
        "Telemedicine Consultation": 80,
        "MRI": 90,
        "X-ray": 70,
        "Blood Test": 60,
    }
    return healthcare_map.get(
        healthcare_needs, 50
    )  # Default score for other needs


def map_insurance_provider(insurance_provider):
    """
    Maps insurance provider to numerical scores.
    """
    insurance_map = {
        "Blue Cross": 90,
        "Aetna": 80,
        "Cigna": 70,
        "UnitedHealthcare": 60,
    }
    return insurance_map.get(
        insurance_provider, 50
    )  # Default score for other providers


def map_provider_referral_trends(referral_trends):
    """
    Maps provider referral trends to numerical scores.
    """
    if referral_trends == "Increasing referrals":
        return 90
    elif referral_trends == "Stable referrals":
        return 70
    elif referral_trends == "Declining referrals":
        return 50
    return 60  # Default case


# Function to calculate Opportunity Score
def calculate_opportunity_score(data):
    weights = {
        "Patient Demographics": 20,
        "Healthcare Needs": 25,
        "Prior Appointment History": 15,
        "Insurance Provider": 20,
        "Provider Referral Trends": 20,
    }

    opportunity_scores = []
    for index, row in data.iterrows():
        demographics_score = map_demographics(row["Patient Demographics"])
        healthcare_needs_score = map_healthcare_needs(row["Healthcare Needs"])
        prior_appointment_score = (
            int(row["Prior Appointment History"]) * 2
        )  # e.g., "3 appointments" -> 3 * 2 = 6
        insurance_provider_score = map_insurance_provider(
            row["Insurance Provider"]
        )
        provider_referral_score = map_provider_referral_trends(
            row["Provider Referral Trends"]
        )

        total_score = (
            (demographics_score * weights["Patient Demographics"])
            + (healthcare_needs_score * weights["Healthcare Needs"])
            + (prior_appointment_score * weights["Prior Appointment History"])
            + (insurance_provider_score * weights["Insurance Provider"])
            + (provider_referral_score * weights["Provider Referral Trends"])
        ) / sum(weights.values())

        opportunity_scores.append(round(total_score, 2))

    return opportunity_scores


# Function to add Opportunity Score to the dataset
def add_opportunity_scores_to_data(data):
    data["Opportunity Score"] = calculate_opportunity_score(data)
    return data


# Advanced Visualizations


def plot_feature_distributions(data):
    """
    Create pair plots and violin plots for feature distributions to understand relationships
    """
    sns.pairplot(
        data[
            [
                "Opportunity Score",
                "Patient Demographics",
                "Healthcare Needs",
                "Prior Appointment History",
                "Insurance Provider",
                "Provider Referral Trends",
            ]
        ]
    )
    plt.suptitle("Feature Distributions and Pairwise Relationships", y=1.02)
    plt.show()

    # Violin Plot to observe distribution of Opportunity Score for different Healthcare Needs
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Healthcare Needs", y="Opportunity Score", data=data, palette="Set2"
    )
    plt.title("Opportunity Score Distribution by Healthcare Needs")
    plt.show()


def plot_correlation_matrix(data):
    corr_data = data[
        [
            "Opportunity Score",
            "Patient Demographics",
            "Healthcare Needs",
            "Prior Appointment History",
            "Insurance Provider",
            "Provider Referral Trends",
        ]
    ].copy()

    # Map categorical data to numerical values
    corr_data["Patient Demographics"] = corr_data[
        "Patient Demographics"
    ].apply(map_demographics)
    corr_data["Healthcare Needs"] = corr_data["Healthcare Needs"].apply(
        map_healthcare_needs
    )
    corr_data["Insurance Provider"] = corr_data["Insurance Provider"].apply(
        map_insurance_provider
    )
    corr_data["Provider Referral Trends"] = corr_data[
        "Provider Referral Trends"
    ].apply(map_provider_referral_trends)

    correlation_matrix = corr_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True
    )
    plt.title("Correlation Matrix for Opportunity Score Prediction")
    plt.show()


# A/B Testing and Model Comparison


def build_and_evaluate_models(data):
    """
    Build and compare multiple models including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
    A/B testing approach by cross-validation and hyperparameter tuning.
    """
    # Features and target variable
    data["Patient Demographics"] = data["Patient Demographics"].apply(
        map_demographics
    )
    data["Healthcare Needs"] = data["Healthcare Needs"].apply(
        map_healthcare_needs
    )
    data["Insurance Provider"] = data["Insurance Provider"].apply(
        map_insurance_provider
    )
    data["Provider Referral Trends"] = data["Provider Referral Trends"].apply(
        map_provider_referral_trends
    )

    features = [
        "Patient Demographics",
        "Healthcare Needs",
        "Prior Appointment History",
        "Insurance Provider",
        "Provider Referral Trends",
    ]
    target = "Opportunity Score"

    X = data[features]
    y = (data[target] >= 75).astype(
        int
    )  # Binary outcome: High Opportunity Score = 1, Low Opportunity Score = 0

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


# Main function to process the data, calculate opportunity scores, visualize, and perform model comparison
def main():
    """
    Main function to read the data from CSV, calculate opportunity scores,
    visualize the data, and perform predictive analysis.
    """
    # Read the existing dataset from the CSV file
    data = pd.read_csv("healthcare_lead_scores.csv")

    # Add the opportunity scores to the data
    data_with_scores = add_opportunity_scores_to_data(data)

    # Visualize the Feature Distributions
    plot_feature_distributions(data_with_scores)

    # Visualize the Correlation Matrix
    plot_correlation_matrix(data_with_scores)

    # Build and evaluate multiple models
    build_and_evaluate_models(data_with_scores)

    # Optionally, save the updated data to a new CSV file
    data_with_scores.to_csv(
        "updated_healthcare_opportunity_scores_with_predictions.csv",
        index=False,
    )
    print(
        "Opportunity scores and predictive analysis have been saved to updated_healthcare_opportunity_scores_with_predictions.csv"
    )


# Execute the main function
if __name__ == "__main__":
    main()
