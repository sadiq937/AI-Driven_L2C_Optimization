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

# Save path for plots
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Lead_score"
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------- Helper Functions --------------------


def map_specialty(specialty):
    return {
        "Cardiology": 80,
        "Neurology": 70,
        "Orthopedics": 60,
        "General Surgery": 50,
        "Pediatrics": 40,
    }.get(specialty, 0)


def map_engagement_history(engagement):
    try:
        return int(engagement.split()[2]) * 2
    except:
        return 0


def map_website_interactions(webtext):
    try:
        return int(webtext.split()[1])
    except:
        return 0


def map_referral_source(referral_source):
    return {
        "Doctor Referral": 80,
        "Digital Ad": 70,
        "Word of Mouth": 60,
        "Online Search": 50,
        "Direct Visit": 40,
    }.get(referral_source, 0)


def map_lead_source(lead_source):
    return {"Digital Ad": 70, "Referral": 80, "Organic Search": 90}.get(
        lead_source, 0
    )


def clean_conversion_probability(prob):
    if isinstance(prob, str):
        prob = prob.strip("%")
    try:
        return float(prob)
    except:
        return 0


# -------------------- Lead Score Calculation --------------------


def calculate_lead_score(data):
    weights = {
        "Specialty": 20,
        "Engagement History": 15,
        "Referral Source": 10,
        "Website Interactions": 15,
        "Lead Source": 10,
        "Conversion Probability": 30,
    }

    raw_scores = []
    for _, row in data.iterrows():
        specialty_score = map_specialty(row["Specialty"])
        engagement_score = map_engagement_history(row["Engagement History"])
        referral_score = map_referral_source(row["Referral Source"])
        website_score = (
            map_website_interactions(row["Website Interactions"]) * 2
        )
        lead_source_score = map_lead_source(row["Lead Source"])
        conv_score = (
            clean_conversion_probability(row["Conversion Probability"]) * 0.3
        )

        total_score = (
            specialty_score * weights["Specialty"]
            + engagement_score * weights["Engagement History"]
            + referral_score * weights["Referral Source"]
            + website_score * weights["Website Interactions"]
            + lead_source_score * weights["Lead Source"]
            + conv_score * weights["Conversion Probability"]
        ) / sum(weights.values())

        raw_scores.append(total_score)

    # Apply Min-Max normalization to 0–100 scale
    min_score = min(raw_scores)
    max_score = max(raw_scores)
    normalized_scores = [
        round(((score - min_score) / (max_score - min_score)) * 100, 2)
        for score in raw_scores
    ]

    return normalized_scores


def add_lead_scores_to_data(data):
    data["Lead Score"] = calculate_lead_score(data)
    return data


# -------------------- Visualizations --------------------


def plot_feature_distributions(data):
    data_plot = data.copy()
    data_plot["Engagement History"] = data_plot["Engagement History"].apply(
        map_engagement_history
    )
    data_plot["Website Interactions"] = data_plot[
        "Website Interactions"
    ].apply(map_website_interactions)

    sns.pairplot(
        data_plot[
            [
                "Lead Score",
                "Engagement History",
                "Website Interactions",
                "Conversion Probability",
            ]
        ]
    )
    plt.suptitle("Feature Distributions", y=1.02)
    plt.savefig(os.path.join(PLOT_DIR, "pairplot_feature_distributions.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Referral Source", y="Lead Score", data=data)
    plt.title("Lead Score by Referral Source")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "violin_leadscore_referral.png"))
    plt.show()


def plot_correlation_matrix(data):
    df_corr = data.copy()
    df_corr["Engagement History"] = df_corr["Engagement History"].apply(
        map_engagement_history
    )
    df_corr["Website Interactions"] = df_corr["Website Interactions"].apply(
        map_website_interactions
    )

    corr_matrix = df_corr[
        [
            "Lead Score",
            "Engagement History",
            "Website Interactions",
            "Conversion Probability",
        ]
    ].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(PLOT_DIR, "correlation_matrix.png"))
    plt.show()


# -------------------- Model Building --------------------


def build_and_evaluate_models(data):
    df_model = data.copy()
    df_model["Referral Source"] = df_model["Referral Source"].apply(
        map_referral_source
    )
    df_model["Lead Source"] = df_model["Lead Source"].apply(map_lead_source)
    df_model["Engagement History"] = df_model["Engagement History"].apply(
        map_engagement_history
    )
    df_model["Website Interactions"] = df_model["Website Interactions"].apply(
        map_website_interactions
    )
    df_model["Conversion Probability"] = df_model[
        "Conversion Probability"
    ].apply(clean_conversion_probability)

    features = [
        "Engagement History",
        "Website Interactions",
        "Referral Source",
        "Lead Source",
        "Conversion Probability",
    ]
    target = "Lead Score"

    X = df_model[features]
    threshold = df_model[target].quantile(0.75)
    print(f"Using dynamic threshold for Lead Score: {threshold}")
    y = (df_model[target] >= threshold).astype(int)
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
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred))

        if hasattr(model, "feature_importances_"):
            sns.barplot(x=features, y=model.feature_importances_)
            plt.title(f"{name} - Feature Importances")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    f"{name.lower().replace(' ', '_')}_feature_importance.png",
                )
            )
            plt.show()

            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            plt.savefig(
                os.path.join(
                    PLOT_DIR,
                    f"{name.lower().replace(' ', '_')}_shap_summary.png",
                ),
                bbox_inches="tight",
            )
            plt.close()

        print("-" * 80)


# -------------------- Main Pipeline --------------------


def main():
    df = pd.read_csv(
        "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv",
        usecols=[
            "Patient ID",
            "Specialty (for providers)",
            "Engagement History",
            "Referral Source",
            "Website Interactions",
            "Lead Source",
            "Conversion Probability",
        ],
    )
    df.rename(columns={"Specialty (for providers)": "Specialty"}, inplace=True)

    df_scored = add_lead_scores_to_data(df)
    df_scored["Conversion Probability"] = (
        df_scored["Conversion Probability"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .astype(float)
    )

    plot_feature_distributions(df_scored)
    plot_correlation_matrix(df_scored)
    build_and_evaluate_models(df_scored)

    # Save back to the same original CSV file (overwrite)
    df.to_csv(
        "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Lead_score.csv",
        index=False,
    )
    print("Lead Score added and file updated successfully!")


if __name__ == "__main__":
    main()
