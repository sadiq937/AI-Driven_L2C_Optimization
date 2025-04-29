import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Paths
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
SAVE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_treatment_conversion_rate.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Treatment_Conversion"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Helper Functions ---

def extract_number(text):
    try:
        return int(text.split()[0])
    except:
        return 0

def map_lead_source(source):
    mapping = {
        "Digital Ad": 70,
        "Referral": 80,
        "Organic Search": 90,
        "Direct Visit": 60,
        "Online Search": 65,
    }
    return mapping.get(source, 50)

# --- Treatment Conversion Rate Calculation ---

def calculate_conversion_rate(df):
    weights = {
        "Lead Source": 25,
        "Consultation Bookings": 25,
        "Insurance Approvals": 25,
        "Follow-up Appointments": 25,
    }

    scores = []
    for _, row in df.iterrows():
        lead_source_score = map_lead_source(row["Lead Source"])
        consultation_score = extract_number(row["Consultation Bookings"])
        insurance_score = extract_number(row["Insurance Approvals"])
        followup_score = extract_number(row["Follow-up Appointments"])

        total_score = (
            lead_source_score * weights["Lead Source"]
            + consultation_score * weights["Consultation Bookings"]
            + insurance_score * weights["Insurance Approvals"]
            + followup_score * weights["Follow-up Appointments"]
        ) / sum(weights.values())

        scores.append(total_score)

    # Normalize to 0–100
    min_score, max_score = min(scores), max(scores)
    normalized = [(s - min_score) / (max_score - min_score) * 100 for s in scores]
    return [round(val, 2) for val in normalized]

# --- Model Building and Evaluation ---

def build_and_evaluate_models(df):
    df["Consultation Bookings"] = df["Consultation Bookings"].apply(extract_number)
    df["Insurance Approvals"] = df["Insurance Approvals"].apply(extract_number)
    df["Follow-up Appointments"] = df["Follow-up Appointments"].apply(extract_number)
    df["Lead Source Encoded"] = df["Lead Source"].apply(map_lead_source)

    features = [
        "Lead Source Encoded",
        "Consultation Bookings",
        "Insurance Approvals",
        "Follow-up Appointments",
    ]
    X = df[features]
    y = (df["Treatment Conversion Rate"] >= df["Treatment Conversion Rate"].quantile(0.75)).astype(int)

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

# --- Visualization ---

def visualize_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Treatment Conversion Rate"], kde=True)
    plt.title("Distribution of Treatment Conversion Rates")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "treatment_conversion_rate_distribution.png"))
    plt.close()

# --- Main Pipeline ---

def main():
    df = pd.read_csv(CSV_PATH, usecols=[
        "Patient ID",
        "Lead Source",
        "Consultation Bookings",
        "Insurance Approvals",
        "Follow-up Appointments",
    ])

    df["Treatment Conversion Rate"] = calculate_conversion_rate(df)

    visualize_distribution(df)
    build_and_evaluate_models(df)

    df.to_csv(SAVE_PATH, index=False)
    print(f"Treatment Conversion Rate calculation complete. File saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
