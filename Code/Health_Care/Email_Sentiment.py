import logging
import os
import re

import lime
import lime.lime_text
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Paths and Logging
CSV_PATH = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/synthetic_healthcare_data.csv"
PLOT_DIR = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Results/Email_Sentiment"
os.makedirs(PLOT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)


def load_data():
    df = pd.read_csv(
        CSV_PATH,
        usecols=[
            "Email Content",
            "Sentiment Analysis",
            "Feedback Rating",
            "Follow-up Response Time",
        ],
    )
    df.dropna(inplace=True)
    df["Cleaned Email"] = df["Email Content"].apply(clean_text)
    df["Feedback Rating"] = (
        df["Feedback Rating"].astype(str).str.extract(r"(\d+)").astype(float)
    )
    df["Follow-up Response Time"] = (
        df["Follow-up Response Time"]
        .astype(str)
        .str.extract(r"(\d+)")
        .astype(float)
    )
    return df


def tfidf_model(X_text, y, label_encoder):
    logging.info("TF-IDF + Traditional ML Models...")
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True),
    }

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"{name} Accuracy: {acc:.4f}")
        print(f"--- {name} Report ---")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0,
            )
        )

        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_,
        )
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                PLOT_DIR,
                f"{name.replace(' ', '_').lower()}_confusion_matrix.png",
            )
        )
        plt.close()
    return model, tfidf


def get_bert_embeddings(texts):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings = []
    for text in tqdm(texts, desc="Encoding BERT"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=64, batch_first=True
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


def train_lstm(X, y):
    logging.info("Training LSTM model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logging.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)
        print("LSTM Accuracy:", acc)
        print(classification_report(y_test, preds))


def evaluate_model(name, y_test, y_pred, label_encoder):
    """Print classification report & save confusion matrix"""
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"{name} Accuracy: {acc:.4f}")
    print(
        f"\n{name} Classification Report:\n",
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        ),
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            PLOT_DIR, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
    )
    plt.close()


def explain_with_shap_and_lime(
    tfidf, model, X_text, X_vec, y, label_encoder, method_name
):
    import lime
    import shap
    from lime.lime_text import LimeTextExplainer

    logging.info(
        f"Explaining predictions using SHAP and LIME for {method_name}..."
    )

    # SHAP for supported models (linear)
    try:
        if method_name in ["logistic_regression", "naive_bayes", "svm"]:
            explainer = shap.Explainer(model.predict_proba, X_vec)
            shap_values = explainer(X_vec[:100])
            shap.summary_plot(
                shap_values,
                features=X_vec[:100],
                feature_names=tfidf.get_feature_names_out(),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(PLOT_DIR, f"{method_name}_shap_summary.png")
            )
            plt.close()
        else:
            logging.warning(f"SHAP not supported for {method_name}")
    except Exception as e:
        logging.warning(f"SHAP explanation failed for {method_name}: {e}")

    # LIME explanation
    try:
        pipeline = make_pipeline(tfidf, model)
        lime_explainer = LimeTextExplainer(class_names=label_encoder.classes_)
        sample_index = 0
        exp = lime_explainer.explain_instance(
            X_text.iloc[sample_index], pipeline.predict_proba, num_features=10
        )
        fig = exp.as_pyplot_figure()
        fig.savefig(
            os.path.join(PLOT_DIR, f"{method_name}_lime_explanation.png")
        )
        plt.close()
    except Exception as e:
        logging.warning(f"LIME explanation failed for {method_name}: {e}")


def model_comparison_bar_chart(results):
    """Bar chart comparing all model accuracies"""
    names = list(results.keys())
    scores = [results[n] for n in names]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=scores, palette="viridis", hue=names, legend=False)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"))
    plt.close()


def run_models(df, label_encoder):
    """Train TF-IDF based models and generate explanations"""
    tfidf = TfidfVectorizer(max_features=1000)
    X_text = df["Cleaned Email"]
    y = label_encoder.transform(df["Sentiment Analysis"])
    X_vec = tfidf.fit_transform(X_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel="linear", probability=True),
    }

    accuracy_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        evaluate_model(name, y_test, y_pred, label_encoder)
        explain_with_shap_and_lime(
            tfidf,
            model,
            X_text,
            X_vec,
            y,
            label_encoder,
            name.lower().replace(" ", "_"),
        )
        accuracy_results[name] = accuracy_score(y_test, y_pred)

    model_comparison_bar_chart(accuracy_results)


def main():
    df = load_data()
    label_encoder = LabelEncoder()
    df["Sentiment Label"] = label_encoder.fit_transform(
        df["Sentiment Analysis"]
    )
    y = df["Sentiment Label"].values

    model, tfidf = tfidf_model(df["Cleaned Email"], y, label_encoder)

    bert_embeddings = get_bert_embeddings(df["Cleaned Email"].tolist())
    X_bert = np.hstack(
        [
            bert_embeddings,
            df[["Feedback Rating", "Follow-up Response Time"]].values,
        ]
    )

    train_lstm(X_bert, y)
    run_models(df, label_encoder)

    # Predict sentiment score
    X_tfidf = tfidf.transform(df["Cleaned Email"])
    positive_class_index = list(label_encoder.classes_).index("Positive")
    df["Email Sentiment Score"] = (
        model.predict_proba(X_tfidf)[:, positive_class_index] * 100
    ).round(2)

    # Save final results
    output_path = "C:/Users/nikhi/Documents/Master_Project/AI-Driven_L2C_Optimization/Data/healthcare_Email_Sentiment_score.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Final sentiment scores saved to {output_path}")


if __name__ == "__main__":
    main()
