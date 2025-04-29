import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
SAVE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_churn_score.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Churn_Score"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Helper Functions ---

def extract_number(text):
    try:
        return int(text.split()[0])
    except:
        return 0

# --- Churn Score Calculation ---

def calculate_churn_score(data):
    weights = {
        "Appointment Frequency": 33.3,
        "Claims Submission History": 33.3,
        "Support Requests": 16.7,
        "Missed Appointments": 16.7,
    }

    scores = []
    for _, row in data.iterrows():
        score = (
            extract_number(row["Appointment Frequency"]) * weights["Appointment Frequency"]
            + extract_number(row["Claims Submission History"]) * weights["Claims Submission History"]
            + extract_number(row["Support Requests"]) * weights["Support Requests"]
            + extract_number(row["Missed Appointments"]) * weights["Missed Appointments"]
        ) / sum(weights.values())

        scores.append(score)

    # Normalize to 0–100
    min_score, max_score = min(scores), max(scores)
    normalized = [(s - min_score) / (max_score - min_score) * 100 for s in scores]
    return [round(val, 2) for val in normalized]

# --- Model Building and Evaluation ---

def build_and_evaluate_models(df):
    df["Appointment Frequency"] = df["Appointment Frequency"].apply(extract_number)
    df["Claims Submission History"] = df["Claims Submission History"].apply(extract_number)
    df["Support Requests"] = df["Support Requests"].apply(extract_number)
    df["Missed Appointments"] = df["Missed Appointments"].apply(extract_number)

    features = [
        "Appointment Frequency",
        "Claims Submission History",
        "Support Requests",
        "Missed Appointments",
    ]
    X = df[features]
    y = (df["Churn Prediction Score"] >= df["Churn Prediction Score"].quantile(0.75)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(8, 5))
            sns.barplot(x=features, y=model.feature_importances_)
            plt.title(f"{name} - Feature Importances")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_').lower()}_feature_importance.png"))
            plt.close()

        try:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_').lower()}_shap_summary.png"))
            plt.close()
        except Exception as e:
            print(f"SHAP failed for {name}: {e}")

# --- Visualizations ---

def visualize_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Churn Prediction Score"], kde=True)
    plt.title("Distribution of Churn Prediction Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "churn_score_distribution.png"))
    plt.close()

# --- Main Pipeline ---

def main():
    df = pd.read_csv(CSV_PATH, usecols=[
        "Patient ID",
        "Appointment Frequency",
        "Claims Submission History",
        "Support Requests",
        "Missed Appointments",
    ])

    df["Churn Prediction Score"] = calculate_churn_score(df)

    visualize_distribution(df)
    build_and_evaluate_models(df)

    df.to_csv(SAVE_PATH, index=False)
    print(f"Churn Score calculation complete. File saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
