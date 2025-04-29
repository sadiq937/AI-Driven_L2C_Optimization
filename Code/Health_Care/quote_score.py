import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Setup paths
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Quote_Score"
os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=[
    "Patient ID", "Treatment Cost Trends", "Insurance Coverage",
    "Hospital Pricing Models", "Government Payer Adjustments"
])

# Normalize and fuzzy map
df["Treatment Cost Trends"] = df["Treatment Cost Trends"].str.lower().str.strip()
df["Insurance Coverage"] = df["Insurance Coverage"].str.lower().str.strip()
df["Hospital Pricing Models"] = df["Hospital Pricing Models"].str.lower().str.strip()
df["Government Payer Adjustments"] = df["Government Payer Adjustments"].str.lower().str.strip()

df["Treatment Cost Trends"] = df["Treatment Cost Trends"].apply(
    lambda x: 1.0 if "increas" in x or "rising" in x else 0.5 if "stable" in x else 0.2 if "declin" in x else np.nan
)
df["Insurance Coverage"] = df["Insurance Coverage"].apply(
    lambda x: 1.0 if "full" in x else 0.5 if "partial" in x else 0.0 if "no" in x else np.nan
)
df["Hospital Pricing Models"] = df["Hospital Pricing Models"].apply(
    lambda x: 1.0 if "premium" in x else 0.5 if "standard" in x else 0.2 if "basic" in x else 0.4 if "fixed" in x else 0.6 if "variable" in x else np.nan
)
df["Government Payer Adjustments"] = df["Government Payer Adjustments"].str.extract(r'(\d+)%').astype(float) / 100

# Clean and check
before = len(df)
df.dropna(inplace=True)
after = len(df)
logging.info(f"Dropped {before - after} rows during mapping cleanup.")

if df.empty:
    logging.error("No data available after preprocessing. Check input categories.")
    exit()

# Quote Generation Score (raw score)
weights = {
    "Treatment Cost Trends": 0.3,
    "Insurance Coverage": 0.25,
    "Hospital Pricing Models": 0.2,
    "Government Payer Adjustments": 0.25,
}
df["Raw Quote Score"] = (
    df["Treatment Cost Trends"] * weights["Treatment Cost Trends"]
    + df["Insurance Coverage"] * weights["Insurance Coverage"]
    + df["Hospital Pricing Models"] * weights["Hospital Pricing Models"]
    + df["Government Payer Adjustments"] * weights["Government Payer Adjustments"]
)

# Normalize to 0-100
minmax = MinMaxScaler(feature_range=(0, 100))
df["Quote Generation Score"] = minmax.fit_transform(df[["Raw Quote Score"]]).round(2)

# Save updated CSV
output_csv = CSV_PATH.replace("synthetic_healthcare_data.csv", "healthcare_quote_generation_score.csv")
df.to_csv(output_csv, index=False)
logging.info(f"Saved updated dataset to {output_csv}")

# Visualizations
plt.figure(figsize=(8, 5))
sns.histplot(df["Quote Generation Score"], kde=True)
plt.title("Distribution of Quote Generation Scores")
plt.savefig(os.path.join(PLOT_DIR, "quote_score_distribution.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Insurance Coverage", y="Quote Generation Score", hue="Hospital Pricing Models")
plt.title("Insurance Coverage vs. Quote Generation Score")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "insurance_vs_quote_score.png"))
plt.close()

# Modeling
features = list(weights.keys())
X = df[features]
y = df["Quote Generation Score"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
}
performance = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    performance[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }

    # SHAP
    try:
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test[:100])
        shap.summary_plot(shap_values, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{name.replace(' ', '_').lower()}_shap_summary.png"))
        plt.close()
    except Exception as e:
        logging.warning(f"SHAP failed for {name}: {e}")

# Neural Network
nn = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1),
])
nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = nn.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=0)

nn_preds = nn.predict(X_test).flatten()
performance["Neural Network"] = {
    "MAE": mean_absolute_error(y_test, nn_preds),
    "MSE": mean_squared_error(y_test, nn_preds),
    "R2": r2_score(y_test, nn_preds),
}

# NN Training Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Neural Network Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "nn_training_loss.png"))
plt.close()

# Save metrics
perf_df = pd.DataFrame(performance).T.round(4)
perf_path = os.path.join(PLOT_DIR, "model_performance_metrics.csv")
perf_df.to_csv(perf_path)
logging.info(f"Performance metrics saved to {perf_path}")
