"""
app.py
------
Flask application that serves the trained Diabetes progression regressor
to the general public, both as:
  - a simple HTML form at "/"              (human-friendly)
  - a JSON REST API at "/api/predict"      (machine-friendly)

Deployed on Render using Docker (see Dockerfile).

Uses real-world units (age in years, actual blood pressure, actual
glucose level) and only 5 fields a person could plausibly know from a
basic health checkup - see train.py for why.
"""

import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "diabetes_model.pkl")
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]

REQUIRED_FIELDS = ["age", "sex", "bmi", "bp", "s6"]

# Realistic example values shown as placeholders in the web form.
EXAMPLE_VALUES = {"age": 45, "sex": 1, "bmi": 26.5, "bp": 92.0, "s6": 95}


def build_features(age, sex, bmi, bp, s6):
    bmi_bp_interaction = bmi * bp
    glucose_bmi_interaction = s6 * bmi
    return np.array([[age, sex, bmi, bp, s6, bmi_bp_interaction, glucose_bmi_interaction]])


def predict_progression(**kwargs):
    X = build_features(**kwargs)
    pred = float(MODEL.predict(X)[0])
    return {"predicted_disease_progression": round(pred, 1)}


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_values = dict(EXAMPLE_VALUES)

    if request.method == "POST":
        try:
            form_values = {k: float(request.form[k]) for k in REQUIRED_FIELDS}
            result = predict_progression(**form_values)
        except (KeyError, ValueError):
            error = "Please enter valid numeric values for all fields."

    return render_template("index.html", result=result, error=error, values=form_values)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for the general public / other programs.

    Fields use real clinical units:
      age  - age in years
      sex  - 1 or 2 (as coded in the original clinical dataset)
      bmi  - body mass index
      bp   - average blood pressure (mm Hg)
      s6   - blood sugar / glucose level (mg/dL)

    Example request:
        POST /api/predict
        {"age": 45, "sex": 1, "bmi": 26.5, "bp": 92.0, "s6": 95}
    """
    data = request.get_json(silent=True) or {}
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        values = {f: float(data[f]) for f in REQUIRED_FIELDS}
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    prediction = predict_progression(**values)
    return jsonify({"input": values, "prediction": prediction})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
