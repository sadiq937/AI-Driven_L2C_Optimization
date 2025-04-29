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

# ------------------ Setup ------------------

LEAD_SCORE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Lead_score.csv"
SYNTHETIC_DATA_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
SAVE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_conversion_probability.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Conversion_Probability"
os.makedirs(PLOT_DIR, exist_ok=True)

# ------------------ Helper Functions ------------------

def extract_number(text):
    try:
        return int(text.split()[0])
    except:
        return 0

def map_referral_source(ref):
    mapping = {
        "Doctor Referral": 80,
        "Digital Ad": 70,
        "Word of Mouth": 60,
        "Online Search": 50,
        "Direct Visit": 40,
    }
    return mapping.get(ref, 50)

def map_insurance_type(ins_type):
    mapping = {
        "Private Insurance": 80,
        "Government Insurance": 70,
        "No Insurance": 40,
    }
    return mapping.get(ins_type, 50)

# ------------------ Score Calculation ------------------

def calculate_conversion_probability(df):
    weights = {
        "Lead Score": 30,
        "Engagement History": 20,
        "Communication History": 20,
        "Referral Source": 15,
        "Insurance Type": 15,
    }

    scores = []
    for _, row in df.iterrows():
        lead_score = row["Lead Score"]
        engagement_score = extract_number(row["Engagement History"])
        communication_score = extract_number(row["Communication History"])
        referral_score = map_referral_source(row["Referral Source"])
        insurance_score = map_insurance_type(row["Insurance Type"])

        total_score = (
            lead_score * weights["Lead Score"]
            + engagement_score * weights["Engagement History"]
            + communication_score * weights["Communication History"]
            + referral_score * weights["Referral Source"]
            + insurance_score * weights["Insurance Type"]
        ) / sum(weights.values())

        scores.append(total_score)

    # Normalize to 0–100
    min_score, max_score = min(scores), max(scores)
    normalized = [(s - min_score) / (max_score - min_score) * 100 for s in scores]
    return [round(val, 2) for val in normalized]

# ------------------ Modeling ------------------

def build_and_evaluate_models(df):
    df["Engagement History"] = df["Engagement History"].apply(extract_number)
    df["Communication History"] = df["Communication History"].apply(extract_number)
    df["Referral Source Encoded"] = df["Referral Source"].apply(map_referral_source)
    df["Insurance Type Encoded"] = df["Insurance Type"].apply(map_insurance_type)

    features = [
        "Lead Score",
        "Engagement History",
        "Communication History",
        "Referral Source Encoded",
        "Insurance Type Encoded",
    ]
    X = df[features]
    y = (df["Conversion Probability"] >= 75).astype(int)  # threshold = 75%

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

# ------------------ Visualization ------------------

def visualize_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Conversion Probability"], kde=True)
    plt.title("Distribution of Conversion Probability Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "conversion_probability_distribution.png"))
    plt.close()

# ------------------ Main Pipeline ------------------

def main():
    # Read Lead Score Data
    df_lead = pd.read_csv(LEAD_SCORE_PATH, usecols=[
        "Patient ID",
        "Lead Score"
    ])

    # Read Synthetic Healthcare Data
    df_health = pd.read_csv(SYNTHETIC_DATA_PATH, usecols=[
        "Patient ID",
        "Engagement History",
        "Communication History",
        "Referral Source",
        "Insurance Type",
    ])

    # Merge on Patient ID
    df = pd.merge(df_lead, df_health, on="Patient ID", how="inner")

    # Calculate Conversion Probability
    df["Conversion Probability"] = calculate_conversion_probability(df)

    # Visualizations
    visualize_distribution(df)
    build_and_evaluate_models(df)

    # Save results
    df.to_csv(SAVE_PATH, index=False)
    print(f"Conversion Probability calculation complete. File saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
