"""
train.py
--------
End-to-end training pipeline for the Heart Disease Prediction project.

Steps performed:
1. Load dataset
2. Clean data (handle missing values, duplicates)
3. Exploratory Data Analysis (EDA) -> graphs saved to static/images/
4. Preprocessing (scaling)
5. Train multiple models: Logistic Regression, Random Forest, SVM, XGBoost
6. Compare accuracy / precision / recall / F1 / ROC-AUC
7. Select the best model
8. Save the best model + scaler + feature list with joblib -> model/

Run:
    python train.py
"""

import os
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, we only save figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Optional model - only used if xgboost is installed
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

TARGET_COL = "target"
RANDOM_STATE = 42

# Columns which are naturally numeric/continuous (rest are categorical/binary)
NUMERIC_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_data(path: str) -> pd.DataFrame:
    """Load the heart disease dataset from CSV."""
    df = pd.read_csv(path)
    print(f"[LOAD] Dataset loaded with shape: {df.shape}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicates.

    Even though this particular cleaned UCI CSV has no missing values,
    real-world exports of this dataset often use '?' for unknown 'ca'
    and 'thal' values, so we defensively coerce and impute anyway -
    this keeps the pipeline robust if you swap in the raw UCI file later.
    """
    df = df.copy()

    # Coerce everything to numeric; anything non-numeric (e.g. '?') becomes NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_before = df.isnull().sum().sum()
    print(f"[CLEAN] Missing values found: {missing_before}")

    # Impute numeric columns with median, categorical/binary with mode
    for col in df.columns:
        if df[col].isnull().any():
            if col in NUMERIC_COLS:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Drop exact duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        print(f"[CLEAN] Dropping {dup_count} duplicate rows")
        df = df.drop_duplicates()

    # Normalize target to strictly 0 / 1 (some UCI variants use 0-4 severity)
    df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if x > 0 else 0)

    print(f"[CLEAN] Final shape after cleaning: {df.shape}")
    return df


def run_eda(df: pd.DataFrame) -> None:
    """Generate and save EDA graphs to static/images/."""
    sns.set_style("whitegrid")

    # 1. Target class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df, hue=TARGET_COL, palette="Set2", legend=False)
    plt.title("Heart Disease Class Distribution (0 = No Disease, 1 = Disease)")
    plt.xlabel("Target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "target_distribution.png"), dpi=120)
    plt.close()

    # 2. Age distribution split by target
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="age", hue=TARGET_COL, kde=True, palette="Set1", bins=20)
    plt.title("Age Distribution by Heart Disease Status")
    plt.xlabel("Age")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "age_distribution.png"), dpi=120)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    # 4. Cholesterol vs Max Heart Rate scatter
    plt.figure(figsize=(7, 4))
    sns.scatterplot(data=df, x="chol", y="thalach", hue=TARGET_COL, palette="Set1")
    plt.title("Cholesterol vs Max Heart Rate")
    plt.xlabel("Cholesterol (mg/dl)")
    plt.ylabel("Max Heart Rate Achieved")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "chol_vs_thalach.png"), dpi=120)
    plt.close()

    # 5. Chest pain type vs target
    plt.figure(figsize=(6, 4))
    sns.countplot(x="cp", hue=TARGET_COL, data=df, palette="Set2")
    plt.title("Chest Pain Type vs Heart Disease")
    plt.xlabel("Chest Pain Type (cp)")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "cp_vs_target.png"), dpi=120)
    plt.close()

    print(f"[EDA] Saved 5 graphs to {IMAGES_DIR}")


def train_models(X_train, X_test, y_train, y_test) -> dict:
    """Train each model and collect evaluation metrics."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    else:
        print("[WARN] xgboost not installed - skipping XGBoost model")

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1_score": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4),
        }
        results[name] = metrics
        trained_models[name] = model

        print(f"[TRAIN] {name}: {metrics}")

    return results, trained_models


def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing accuracy of each model."""
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, accuracies, color=sns.color_palette("Set2", len(names)))
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.2%}",
                  ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "model_comparison.png"), dpi=120)
    plt.close()
    print("[COMPARE] Saved model_comparison.png")


def main():
    df = load_data(DATASET_PATH)
    df = clean_data(df)
    run_eda(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale features - fit ONLY on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    plot_model_comparison(results)

    # Pick best model by accuracy (ties broken by ROC-AUC)
    best_name = max(results, key=lambda n: (results[n]["accuracy"], results[n]["roc_auc"]))
    best_model = trained_models[best_name]
    print(f"\n[BEST MODEL] {best_name} -> {results[best_name]}")

    # Save confusion matrix of the best model
    preds = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title(f"Confusion Matrix - {best_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Persist artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics": results[best_name],
            "all_results": results,
            "feature_names": feature_names,
        }, f, indent=2)

    print(f"\n[SAVE] Model, scaler, feature list, and model_info.json saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
