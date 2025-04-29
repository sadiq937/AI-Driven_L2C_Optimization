import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

# Save directory for plots
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Opportunity_score"
os.makedirs(PLOT_DIR, exist_ok=True)


# ------------------ Mapping Functions ------------------ #
def map_demographics(demo):
    """
    Maps demographics to numerical values.
    Example: Age group and gender mapped to values.
    """
    try:
        gender, age, state = demo.split(", ")
        score = 0
        score += 10 if gender.strip() == "Male" else 15
        score += int(age.strip()) * 0.5
        state_map = {"California": 20, "New York": 18, "Texas": 15}
        score += state_map.get(state.strip(), 10)
        return round(score, 2)
    except:
        return 10


def map_healthcare_needs(need):
    """
    Maps healthcare needs (e.g., services required) to numerical scores.
    """
    mapping = {
        "Telemedicine Consultation": 80,
        "Routine Checkup": 70,
        "MRI Scan": 90,
        "Surgery Consultation": 85,
        "Emergency Care": 95,
    }
    return mapping.get(need, 50)


def map_prior_appointment_history(history):
    try:
        count = int(history.split(" ")[0])
        return count * 10
    except:
        return 0


def map_insurance_provider(provider):
    """
    Maps insurance provider to numerical scores.
    """
    mapping = {
        "Blue Cross Blue Shield": 90,
        "Aetna": 80,
        "Cigna": 75,
        "United Healthcare": 70,
        "Humana": 65,
    }
    return mapping.get(provider, 50)


def map_referral_trends(trend):
    """
    Maps provider referral trends to numerical scores.
    """
    mapping = {"Increasing": 90, "Stable": 70, "Decreasing": 50}
    return mapping.get(trend, 60)


# ------------------ Score Calculation ------------------ #
def calculate_opportunity_score(df):
    weights = {
        "Patient Demographics": 20,
        "Healthcare Needs": 25,
        "Prior Appointment History": 15,
        "Insurance Provider": 20,
        "Provider Referral Trends": 20,
    }

    scores = []
    for _, row in df.iterrows():
        demo_score = map_demographics(row["Patient Demographics"])
        needs_score = map_healthcare_needs(row["Healthcare Needs"])
        appt_score = map_prior_appointment_history(
            row["Prior Appointment History"]
        )
        insurance_score = map_insurance_provider(row["Insurance Provider"])
        trend_score = map_referral_trends(row["Provider Referral Trends"])

        total = (
            demo_score * weights["Patient Demographics"]
            + needs_score * weights["Healthcare Needs"]
            + appt_score * weights["Prior Appointment History"]
            + insurance_score * weights["Insurance Provider"]
            + trend_score * weights["Provider Referral Trends"]
        ) / sum(weights.values())
        scores.append(round(total, 2))
    return scores


# ------------------ Visualization ------------------ #
def plot_feature_distributions(df):
    """
    Create pair plots and violin plots for feature distributions to understand relationships
    """
    df_plot = df.copy()
    df_plot["Patient Demographics"] = df_plot["Patient Demographics"].apply(
        map_demographics
    )
    df_plot["Healthcare Needs"] = df_plot["Healthcare Needs"].apply(
        map_healthcare_needs
    )
    df_plot["Prior Appointment History"] = df_plot[
        "Prior Appointment History"
    ].apply(map_prior_appointment_history)
    df_plot["Insurance Provider"] = df_plot["Insurance Provider"].apply(
        map_insurance_provider
    )
    df_plot["Provider Referral Trends"] = df_plot[
        "Provider Referral Trends"
    ].apply(map_referral_trends)

    sns.pairplot(
        df_plot[
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
    plt.suptitle("Feature Distributions", y=1.02)
    plt.savefig(f"{PLOT_DIR}/pairplot_opportunity_score.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Healthcare Needs", y="Opportunity Score", data=df)
    plt.title("Opportunity Score by Healthcare Needs")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/violin_healthcare_needs.png")
    plt.close()


def plot_correlation_matrix(df):
    df_corr = df.copy()
    df_corr["Patient Demographics"] = df_corr["Patient Demographics"].apply(
        map_demographics
    )
    df_corr["Healthcare Needs"] = df_corr["Healthcare Needs"].apply(
        map_healthcare_needs
    )
    df_corr["Prior Appointment History"] = df_corr[
        "Prior Appointment History"
    ].apply(map_prior_appointment_history)
    df_corr["Insurance Provider"] = df_corr["Insurance Provider"].apply(
        map_insurance_provider
    )
    df_corr["Provider Referral Trends"] = df_corr[
        "Provider Referral Trends"
    ].apply(map_referral_trends)

    corr = df_corr[
        [
            "Opportunity Score",
            "Patient Demographics",
            "Healthcare Needs",
            "Prior Appointment History",
            "Insurance Provider",
            "Provider Referral Trends",
        ]
    ].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix - Opportunity Score")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/correlation_matrix.png")
    plt.close()


# ------------------ Model Evaluation ------------------ #
def build_and_evaluate_models(df):
    """
    Build and compare multiple models including Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
    A/B testing approach by cross-validation and hyperparameter tuning.
    """
    df_model = df.copy()
    df_model["Patient Demographics"] = df_model["Patient Demographics"].apply(
        map_demographics
    )
    df_model["Healthcare Needs"] = df_model["Healthcare Needs"].apply(
        map_healthcare_needs
    )
    df_model["Prior Appointment History"] = df_model[
        "Prior Appointment History"
    ].apply(map_prior_appointment_history)
    df_model["Insurance Provider"] = df_model["Insurance Provider"].apply(
        map_insurance_provider
    )
    df_model["Provider Referral Trends"] = df_model[
        "Provider Referral Trends"
    ].apply(map_referral_trends)

    features = [
        "Patient Demographics",
        "Healthcare Needs",
        "Prior Appointment History",
        "Insurance Provider",
        "Provider Referral Trends",
    ]
    target = "Opportunity Score"
    threshold = df_model[target].quantile(0.75)
    y = (df_model[target] >= threshold).astype(int)

    print("Target Class Distribution:")
    print(y.value_counts())

    X = df_model[features]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(8, 4))
            sns.barplot(x=features, y=model.feature_importances_)
            plt.title(f"{name} Feature Importance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                f"{PLOT_DIR}/{name.lower().replace(' ', '_')}_feature_importance.png"
            )
            plt.close()

            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(
                f"{PLOT_DIR}/{name.lower().replace(' ', '_')}_shap_summary.png"
            )
            plt.close()


# ------------------ Main Script ------------------ #
def main():
    """
    Main function to read the data from CSV, calculate opportunity scores,
    visualize the data, and perform predictive analysis.
    """
    df = pd.read_csv(
        "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
    )

    cols_needed = [
        "Patient Demographics",
        "Healthcare Needs",
        "Prior Appointment History",
        "Insurance Provider",
        "Provider Referral Trends",
    ]
    df = df[cols_needed].copy()
    df["Opportunity Score"] = calculate_opportunity_score(df)

    plot_feature_distributions(df)
    plot_correlation_matrix(df)
    build_and_evaluate_models(df)

    # Save back to the same original CSV file (overwrite)
    df.to_csv(
        "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Opportunity_Score.csv",
        index=False,
    )
    print("Lead Score added and file updated successfully!")


if __name__ == "__main__":
    main()
