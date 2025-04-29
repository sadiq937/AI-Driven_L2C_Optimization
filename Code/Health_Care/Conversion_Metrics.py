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
SAVE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_conversion_metrics.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Conversion_Metrics"
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Helper Functions ---

def map_patient_acquisition(value):
    val = str(value).lower()
    if "10% increase" in val:
        return 70.0
    elif "stable" in val:
        return 50.0
    elif "decline" in val:
        return 30.0
    else:
        return 40.0  # default fallback

def extract_percentage(text):
    try:
        return float("".join(filter(str.isdigit, str(text))))
    except:
        return 0.0

# --- Conversion Metrics Calculation ---

def calculate_conversion_metrics(df):
    weights = {
        "Patient Acquisition Rate": 40,
        "Marketing Channel Performance": 30,
        "Referral Success Rate": 30,
    }

    scores = []
    for _, row in df.iterrows():
        acquisition = map_patient_acquisition(row["Patient Acquisition Rate"])
        marketing = extract_percentage(row["Marketing Channel Performance"])
        referral = extract_percentage(row["Referral Success Rate"])

        total_score = (
            acquisition * weights["Patient Acquisition Rate"]
            + marketing * weights["Marketing Channel Performance"]
            + referral * weights["Referral Success Rate"]
        ) / sum(weights.values())

        scores.append(total_score)

    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        print("Warning: No variation in scores. Defaulting to 50.0.")
        return [50.0 for _ in scores]

    normalized = [
        round(((s - min_score) / (max_score - min_score)) * 100, 2) for s in scores
    ]
    return normalized

# --- Modeling & Evaluation ---

def build_and_evaluate_models(df):
    df["Patient Acquisition Rate"] = df["Patient Acquisition Rate"].apply(map_patient_acquisition)
    df["Marketing Channel Performance"] = df["Marketing Channel Performance"].apply(extract_percentage)
    df["Referral Success Rate"] = df["Referral Success Rate"].apply(extract_percentage)

    features = [
        "Patient Acquisition Rate",
        "Marketing Channel Performance",
        "Referral Success Rate",
    ]
    X = df[features]
    y = (df["Conversion Metrics"] >= 35).astype(int)

    print("\nClass distribution:\n", y.value_counts())

    if len(np.unique(y)) < 2:
        print("⚠️ Only one class in target. Skipping model training.")
        return

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
    sns.histplot(df["Conversion Metrics"], kde=True)
    plt.title("Distribution of Conversion Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "conversion_metrics_distribution.png"))
    plt.close()

# --- Main ---

def main():
    df = pd.read_csv(CSV_PATH, usecols=[
        "Patient ID",
        "Patient Acquisition Rate",
        "Marketing Channel Performance",
        "Referral Success Rate",
    ])

    df["Conversion Metrics"] = calculate_conversion_metrics(df)

    visualize_distribution(df)
    build_and_evaluate_models(df)

    df.to_csv(SAVE_PATH, index=False)
    print(f"Conversion Metrics calculated and saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()
