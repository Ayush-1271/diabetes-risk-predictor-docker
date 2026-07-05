"""
train.py
--------
Trains a regressor on the classic Diabetes dataset (Efron et al., 2004),
built directly into scikit-learn - predicting a quantitative measure of
disease progression one year after baseline.

Two deliberate choices make this deployment genuinely usable by a
non-technical person, unlike a naive use of this dataset:

1. Loaded with scaled=False, which returns the ORIGINAL real-world-unit
   values (age in years, actual blood pressure, actual glucose level,
   etc.) instead of scikit-learn's default pre-standardized z-scores that
   nobody outside a stats class could meaningfully fill into a form.
   StandardScaler inside the pipeline handles normalization internally.

2. Reduced to 5 features a person could plausibly know from a basic
   health checkup - age, sex, BMI, average blood pressure, and blood
   sugar (glucose) level - dropping the 4 detailed lipid-panel values
   (total cholesterol, LDL, HDL, cholesterol/HDL ratio, triglycerides)
   that require lab results most people don't have on hand. Blood sugar
   in particular is directly relevant to a DIABETES predictor.

Pipeline:
    1. Load data (raw units)
    2. EDA (saved as PNG plots)
    3. Feature engineering (derived interaction features)
    4. Train/test split
    5. Hyperparameter tuning (GridSearchCV) on a RandomForestRegressor
       wrapped in a scikit-learn Pipeline (StandardScaler + model)
    6. Evaluation (RMSE, MAE, R^2)
    7. Save the fitted pipeline to diabetes_model.pkl
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Only the features a general user could plausibly provide from a basic
# checkup - dropping the detailed lipid panel (s1-s5) on purpose.
BASE_FEATURES = ["age", "sex", "bmi", "bp", "s6"]
ENGINEERED_FEATURES = ["bmi_bp_interaction", "glucose_bmi_interaction"]
FEATURE_NAMES = BASE_FEATURES + ENGINEERED_FEATURES


def load_data() -> pd.DataFrame:
    # scaled=False returns real-world units (age in years, actual blood
    # pressure, actual glucose level, etc.) instead of pre-standardized
    # z-scores, so the web form can ask for numbers a person actually has.
    data = load_diabetes(as_frame=True, scaled=False)
    df = data.frame.rename(columns={"target": "disease_progression"})
    return df


def run_eda(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 5))
    corr = df[BASE_FEATURES + ["disease_progression"]].corr()
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", annot_kws={"size": 8})
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    plt.savefig("eda_correlation_heatmap.png", dpi=110)
    plt.close("all")

    plt.figure(figsize=(6, 4))
    sns.histplot(df["disease_progression"], bins=30, kde=True)
    plt.title("Distribution of disease progression score")
    plt.tight_layout()
    plt.savefig("eda_target_distribution.png", dpi=110)
    plt.close("all")

    print("EDA plots saved: eda_correlation_heatmap.png, eda_target_distribution.png")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Real physical interaction terms, now that features are raw units."""
    df = df.copy()
    df["bmi_bp_interaction"] = df["bmi"] * df["bp"]
    df["glucose_bmi_interaction"] = df["s6"] * df["bmi"]
    return df


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )


def train_and_tune(X_train, y_train) -> GridSearchCV:
    pipeline = build_pipeline()
    param_grid = {
        "reg__n_estimators": [80, 150],
        "reg__max_depth": [3, 5, None],
        "reg__min_samples_leaf": [1, 3, 5],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, preds, alpha=0.5, s=18)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual disease progression")
    plt.ylabel("Predicted disease progression")
    plt.title(f"Predicted vs Actual (R2={r2:.3f})")
    plt.tight_layout()
    plt.savefig("eda_predicted_vs_actual.png", dpi=110)
    plt.close("all")

    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    print("Loading data...")
    df = load_data()

    print("Running EDA...")
    run_eda(df)

    print("Engineering features...")
    df = engineer_features(df)

    X = df[FEATURE_NAMES]
    y = df["disease_progression"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("Tuning hyperparameters with GridSearchCV...")
    grid = train_and_tune(X_train, y_train)
    best_model = grid.best_estimator_
    print("Best params:", grid.best_params_)
    print("Best CV R^2: %.4f" % grid.best_score_)

    print("Evaluating on held-out test set...")
    metrics = evaluate(best_model, X_test, y_test)
    print(json.dumps(metrics, indent=2))

    print("Refitting best pipeline on full dataset...")
    final_model = build_pipeline().set_params(**grid.best_params_)
    final_model.fit(X, y)

    joblib.dump(
        {"model": final_model, "feature_names": FEATURE_NAMES},
        "diabetes_model.pkl",
    )
    print("Saved trained pipeline to diabetes_model.pkl")


if __name__ == "__main__":
    main()
