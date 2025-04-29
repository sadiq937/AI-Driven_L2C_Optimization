import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# -------------------- Paths --------------------
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
SAVE_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_discount_score.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Discount_Score"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------- Helper Functions --------------------
def map_insurance_coverage(coverage):
    coverage = str(coverage).lower()
    if "full" in coverage:
        return 1.0
    elif "partial" in coverage:
        return 0.5
    elif "no" in coverage:
        return 0.0
    return 0.0

def map_payment_history(payment):
    payment = str(payment).lower()
    if "paid in full" in payment:
        return 1.0
    elif "pending" in payment:
        return 0.5
    elif "overdue" in payment:
        return 0.2
    return 0.0

def map_referral_trends(referral):
    referral = str(referral).lower()
    if "increas" in referral:
        return 1.0
    elif "stable" in referral:
        return 0.6
    elif "decreas" in referral:
        return 0.3
    return 0.0

def map_upselling_potential(upsell):
    upsell = str(upsell).lower()
    if "high" in upsell:
        return 1.0
    elif "moderate" in upsell or "medium" in upsell:
        return 0.6
    elif "low" in upsell:
        return 0.3
    return 0.0

# -------------------- Discount Score Calculation --------------------
def calculate_discount_score(df):
    weights = {
        "Insurance Coverage": 30,
        "Payment History": 25,
        "Provider Referral Trends": 25,
        "Upselling Potential": 20,
    }

    df["Insurance Score"] = df["Insurance Coverage"].apply(map_insurance_coverage)
    df["Payment Score"] = df["Payment History"].apply(map_payment_history)
    df["Referral Score"] = df["Provider Referral Trends"].apply(map_referral_trends)
    df["Upsell Score"] = df["Upselling Potential"].apply(map_upselling_potential)

    raw_scores = (
        df["Insurance Score"] * weights["Insurance Coverage"]
        + df["Payment Score"] * weights["Payment History"]
        + df["Referral Score"] * weights["Provider Referral Trends"]
        + df["Upsell Score"] * weights["Upselling Potential"]
    ) / sum(weights.values()) * 100

    df["Discount Score"] = raw_scores.round(2)

    df["Discount Recommendation"] = df["Discount Score"].apply(
        lambda x: "20% Discount Recommended" if x >= 80 else
                  "10% Discount Recommended" if x >= 60 else
                  "5% Discount Recommended" if x >= 40 else
                  "No Discount Recommended"
    )

    return df

# -------------------- Modeling --------------------
def build_and_evaluate_models(df):
    features = ["Insurance Score", "Payment Score", "Referral Score", "Upsell Score"]
    X = df[features]
    y = (df["Discount Score"] >= df["Discount Score"].quantile(0.75)).astype(int)

    print("\nClass distribution:\n", y.value_counts())

    if len(np.unique(y)) < 2:
        print("Warning: Only one class present in data. Skipping model training.")
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

# -------------------- Visualization --------------------
def visualize_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Discount Score"], kde=True)
    plt.title("Distribution of Discount Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "discount_score_distribution.png"))
    plt.close()

# -------------------- Main Pipeline --------------------
def main():
    df = pd.read_csv(CSV_PATH, usecols=[
        "Patient ID",
        "Insurance Coverage",
        "Payment History",
        "Provider Referral Trends",
        "Upselling Potential"
    ])

    df = calculate_discount_score(df)
    visualize_distribution(df)
    build_and_evaluate_models(df)
    df.to_csv(SAVE_PATH, index=False)
    print(f"Discount Score calculated and saved to: {SAVE_PATH}")

if __name__ == "__main__":
    main()