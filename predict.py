"""
predict.py
----------
Loads the trained model/scaler/feature list produced by train.py and
exposes a single function, `predict_heart_disease`, that Flask (app.py)
calls to turn raw form input into a prediction.

Keeping this logic separate from app.py means:
- The model-loading code is only ever written once.
- app.py stays focused on routing/HTTP concerns.
- This file could be unit-tested or reused (e.g. in a CLI or batch script)
  without needing Flask at all.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")

# Human-readable descriptions used both for validation and for the frontend
FEATURE_SPECS = {
    "age":      {"label": "Age",                          "min": 1,   "max": 120,  "type": float},
    "sex":      {"label": "Sex (1 = male, 0 = female)",    "min": 0,   "max": 1,    "type": float},
    "cp":       {"label": "Chest Pain Type (0-3)",         "min": 0,   "max": 3,    "type": float},
    "trestbps": {"label": "Resting Blood Pressure",        "min": 60,  "max": 250,  "type": float},
    "chol":     {"label": "Serum Cholesterol (mg/dl)",     "min": 100, "max": 700,  "type": float},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl (1 = yes, 0 = no)", "min": 0, "max": 1, "type": float},
    "restecg":  {"label": "Resting ECG Results (0-2)",     "min": 0,   "max": 2,    "type": float},
    "thalach":  {"label": "Max Heart Rate Achieved",       "min": 60,  "max": 250,  "type": float},
    "exang":    {"label": "Exercise Induced Angina (1 = yes, 0 = no)", "min": 0, "max": 1, "type": float},
    "oldpeak":  {"label": "ST Depression Induced by Exercise", "min": 0, "max": 10, "type": float},
    "slope":    {"label": "Slope of Peak Exercise ST Segment (0-2)", "min": 0, "max": 2, "type": float},
    "ca":       {"label": "Number of Major Vessels Colored (0-4)", "min": 0, "max": 4, "type": float},
    "thal":     {"label": "Thalassemia (0-3)",             "min": 0,   "max": 3,    "type": float},
}


class ModelNotFoundError(Exception):
    """Raised when model artifacts haven't been trained/saved yet."""
    pass


def _load_artifacts():
    """Load model, scaler, and feature list from disk. Raises a clear error if missing."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(FEATURES_PATH)):
        raise ModelNotFoundError(
            "Model artifacts not found. Run 'python train.py' first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, scaler, feature_names


# Load once at import time so every request doesn't re-read from disk.
try:
    _model, _scaler, _feature_names = _load_artifacts()
    MODEL_LOAD_ERROR = None
except ModelNotFoundError as e:
    _model, _scaler, _feature_names = None, None, None
    MODEL_LOAD_ERROR = str(e)


def validate_input(data: dict):
    """
    Validate raw input dict against FEATURE_SPECS.
    Returns (cleaned_dict, list_of_errors).
    """
    cleaned = {}
    errors = []

    for feature, spec in FEATURE_SPECS.items():
        raw_value = data.get(feature)

        if raw_value is None or str(raw_value).strip() == "":
            errors.append(f"'{spec['label']}' is required.")
            continue

        try:
            value = spec["type"](raw_value)
        except (ValueError, TypeError):
            errors.append(f"'{spec['label']}' must be a number.")
            continue

        if not (spec["min"] <= value <= spec["max"]):
            errors.append(
                f"'{spec['label']}' must be between {spec['min']} and {spec['max']}."
            )
            continue

        cleaned[feature] = value

    return cleaned, errors


def predict_heart_disease(data: dict):
    """
    Run a full prediction cycle:
    1. Validate input
    2. Order features to match training order
    3. Scale
    4. Predict class + probability

    Returns a dict:
        {
          "success": bool,
          "errors": [...],            # only if success is False
          "prediction": 0 or 1,       # only if success is True
          "label": "Disease" | "No Disease",
          "probability": float (0-1),# probability of the predicted class
          "risk_percent": float       # probability of disease specifically
        }
    """
    if _model is None:
        return {"success": False, "errors": [MODEL_LOAD_ERROR]}

    cleaned, errors = validate_input(data)
    if errors:
        return {"success": False, "errors": errors}

    # Build feature vector in the exact order the model was trained on
    try:
        ordered_values = [cleaned[feat] for feat in _feature_names]
    except KeyError as e:
        return {"success": False, "errors": [f"Missing expected feature: {e}"]}

    X = pd.DataFrame([ordered_values], columns=_feature_names)
    X_scaled = _scaler.transform(X)

    pred_class = int(_model.predict(X_scaled)[0])
    proba = _model.predict_proba(X_scaled)[0]
    disease_probability = float(proba[1])

    return {
        "success": True,
        "prediction": pred_class,
        "label": "Disease Detected" if pred_class == 1 else "No Disease Detected",
        "probability": round(float(proba[pred_class]), 4),
        "risk_percent": round(disease_probability * 100, 2),
    }
