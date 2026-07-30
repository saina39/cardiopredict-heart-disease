"""
app.py
------
Flask web application for the Heart Disease Prediction project.

Routes:
    GET  /                 -> Home page
    GET  /predict          -> Prediction form
    POST /predict          -> Handles form submission, renders result page
    POST /api/predict      -> JSON API endpoint (for programmatic access / JS fetch)
    GET  /about            -> Model info / accuracy comparison page
    404 / 500 handlers     -> Friendly error pages
"""

import os
import json
import logging

from flask import Flask, render_template, request, jsonify

from predict import predict_heart_disease, FEATURE_SPECS, MODEL_LOAD_ERROR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_INFO_PATH = os.path.join(BASE_DIR, "model", "model_info.json")

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_info():
    """Read model_info.json (created by train.py) for the About/Model page."""
    if not os.path.exists(MODEL_INFO_PATH):
        return None
    with open(MODEL_INFO_PATH, "r") as f:
        return json.load(f)


@app.route("/")
def home():
    """Landing page."""
    model_info = load_model_info()
    return render_template("index.html", model_info=model_info)


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    """
    GET  -> show the prediction form
    POST -> validate input, run prediction, show result page
    """
    if request.method == "GET":
        return render_template("predict.html", features=FEATURE_SPECS, form_data={})

    # POST: gather form data
    form_data = request.form.to_dict()

    try:
        result = predict_heart_disease(form_data)
    except Exception as exc:  # noqa: BLE001 - want to surface any prediction error safely
        logger.exception("Unexpected error during prediction")
        return render_template(
            "predict.html",
            features=FEATURE_SPECS,
            form_data=form_data,
            errors=[f"An unexpected error occurred: {exc}"],
        )

    if not result.get("success"):
        return render_template(
            "predict.html",
            features=FEATURE_SPECS,
            form_data=form_data,
            errors=result.get("errors", ["Prediction failed."]),
        )

    return render_template("result.html", result=result, form_data=form_data)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint.

    Example request:
        POST /api/predict
        Content-Type: application/json
        {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
        }
    """
    if not request.is_json:
        return jsonify({"success": False, "errors": ["Request must be JSON."]}), 400

    data = request.get_json(silent=True) or {}

    try:
        result = predict_heart_disease(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in /api/predict")
        return jsonify({"success": False, "errors": [str(exc)]}), 500

    status_code = 200 if result.get("success") else 400
    return jsonify(result), status_code


@app.route("/about")
def about():
    """Shows model comparison metrics and which model was selected."""
    model_info = load_model_info()
    return render_template("about.html", model_info=model_info)


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Something went wrong on our end."), 500


if __name__ == "__main__":
    if MODEL_LOAD_ERROR:
        logger.warning(MODEL_LOAD_ERROR)
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
