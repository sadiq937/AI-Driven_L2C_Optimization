import logging
import os

import matplotlib.pyplot as plt
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

# Paths
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
ENGAGEMENT_PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Engagement_Score"
os.makedirs(ENGAGEMENT_PLOT_DIR, exist_ok=True)

# --------------------- Mappers ---------------------


def map_telehealth(value):
    return {"Low": 20, "Medium": 50, "High": 80}.get(value, 0)


def map_appointment_freq(value):
    return {"1 per year": 20, "2 per year": 40, "Quarterly": 80}.get(value, 0)


def extract_support_requests(value):
    try:
        return int(value.split()[0])
    except:
        return 0


def extract_portal_usage(value):
    try:
        return int(value.split()[2])  # "Logged in 5 times"
    except:
        return 0


# --------------------- Engagement Score ---------------------


def calculate_engagement_score(data):
    """
    Calculates engagement scores based on healthcare interactions.
    """
    weights = {
        "Telehealth Engagement": 0.3,
        "Appointment Frequency": 0.3,
        "Support Requests": 0.2,
        "Patient Portal Usage": 0.2,
    }

    data["Telehealth Score"] = data["Telehealth Engagement"].apply(
        map_telehealth
    )
    data["Appointment Score"] = data["Appointment Frequency"].apply(
        map_appointment_freq
    )
    data["Support Score"] = data["Support Requests"].apply(
        extract_support_requests
    )
    data["Portal Score"] = data["Patient Portal Usage"].apply(
        extract_portal_usage
    )

    data["Engagement Score"] = (
        data["Telehealth Score"] * weights["Telehealth Engagement"]
        + data["Appointment Score"] * weights["Appointment Frequency"]
        + data["Support Score"] * weights["Support Requests"]
        + data["Portal Score"] * weights["Patient Portal Usage"]
    ).round(2)

    return data.drop(
        columns=[
            "Telehealth Score",
            "Appointment Score",
            "Support Score",
            "Portal Score",
        ]
    )


# --------------------- Visualizations ---------------------


def plot_feature_distributions(data):
    """
    Plots and saves feature distributions as pairplots and violin plots.
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
    plt.savefig(
        os.path.join(ENGAGEMENT_PLOT_DIR, "feature_distributions_pairplot.png")
    )
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Patient Portal Usage",
        y="Engagement Score",
        hue="Patient Portal Usage",
        data=data,
        palette="coolwarm",
        legend=False,
    )

    plt.title("Engagement Score Distribution by Patient Portal Usage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(
        os.path.join(ENGAGEMENT_PLOT_DIR, "violin_patient_portal_usage.png")
    )
    plt.close()


def plot_correlation_matrix(data):
    """
    Plots and saves correlation matrix of numerical features only.
    """
    corr_data = data.copy()
    corr_data["Support Requests"] = corr_data["Support Requests"].apply(
        extract_support_requests
    )
    corr_data["Patient Portal Usage"] = corr_data[
        "Patient Portal Usage"
    ].apply(extract_portal_usage)
    corr_data["Appointment Frequency"] = corr_data[
        "Appointment Frequency"
    ].apply(map_appointment_freq)
    corr_data["Telehealth Engagement"] = corr_data[
        "Telehealth Engagement"
    ].apply(map_telehealth)

    numeric_cols = [
        "Telehealth Engagement",
        "Appointment Frequency",
        "Support Requests",
        "Patient Portal Usage",
        "Engagement Score",
    ]

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_data[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f"
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(ENGAGEMENT_PLOT_DIR, "correlation_matrix.png"))
    plt.close()


# --------------------- Model Building ---------------------


def build_and_evaluate_models(data):
    """
    Trains and evaluates models for Engagement Score classification.
    """
    features = [
        "Telehealth Engagement",
        "Appointment Frequency",
        "Support Requests",
        "Patient Portal Usage",
    ]

    df = data.copy()
    df["Telehealth Engagement"] = df["Telehealth Engagement"].apply(
        map_telehealth
    )
    df["Appointment Frequency"] = df["Appointment Frequency"].apply(
        map_appointment_freq
    )
    df["Support Requests"] = df["Support Requests"].apply(
        extract_support_requests
    )
    df["Patient Portal Usage"] = df["Patient Portal Usage"].apply(
        extract_portal_usage
    )

    target = "Engagement Score"
    X = df[features]

    # Debug scoring
    print("Engagement Score stats:\n", df[target].describe())

    # Dynamic threshold (e.g., top 25% are high engagement)
    threshold = df[target].quantile(0.75)
    print(f"Using threshold: {threshold}")
    y = (df[target] >= threshold).astype(int)

    print("Target class distribution:\n", y.value_counts())

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
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(50, 30), max_iter=500, random_state=42
        ),
    }

    for name, model in models.items():
        logging.info(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(8, 4))
            sns.barplot(x=features, y=model.feature_importances_)
            plt.title(f"Feature Importance - {name}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    ENGAGEMENT_PLOT_DIR, f"{name}_feature_importance.png"
                )
            )
            plt.close()

            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(
                os.path.join(ENGAGEMENT_PLOT_DIR, f"{name}_shap_summary.png")
            )
            plt.close()

        print("-" * 80)


# --------------------- Main ---------------------


def main():
    """
    Loads data, calculates engagement score, saves updated file,
    creates plots, and evaluates models.
    """
    try:
        data = pd.read_csv(CSV_PATH)
        logging.info("Dataset loaded.")

        data = calculate_engagement_score(data)
        logging.info("Engagement Score calculated.")

        data.to_csv(
            "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Engagement_score.csv",
            index=False,
        )
        logging.info(f"Dataset updated with Engagement Score at: {CSV_PATH}")

        plot_feature_distributions(data)
        plot_correlation_matrix(data)
        build_and_evaluate_models(data)

    except Exception as e:
        logging.error(f"Error: {e}")


if __name__ == "__main__":
    main()
