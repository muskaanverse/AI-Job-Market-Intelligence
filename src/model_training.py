"""
model_training.py
-----------------
Trains 4 regression models, compares them, and saves the best one.

MODELS COMPARED:
1. Linear Regression  — The baseline. Simple and interpretable.
                        If a fancy model can't beat it, something's wrong.
2. Decision Tree      — Learns non-linear rules. Easy to visualize.
                        Tends to overfit without limits.
3. Random Forest      — An ensemble of 100+ decision trees.
                        More stable than a single tree. Usually strong.
4. XGBoost            — Gradient boosting. Wins most Kaggle competitions.
                        The current industry standard for tabular data.

WHY CROSS-VALIDATION?
A single train/test split can get lucky (or unlucky) depending on the random split.
5-fold CV trains the model 5 times on different splits and averages the scores.
This gives a much more reliable estimate of real-world performance.
Recruiters look for this — it shows you understand proper ML evaluation.

METRICS EXPLAINED:
- R²   : How much variance does the model explain? 1.0 = perfect, 0 = no better than guessing the mean.
- RMSE : Root Mean Squared Error — average prediction error in dollars (penalizes large errors).
- MAE  : Mean Absolute Error — simpler average error in dollars (easier to explain to stakeholders).
"""

import os
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")


def get_models() -> dict:
    """
    Returns the 4 models with sensible default hyperparameters.
    random_state=42 ensures reproducibility — every run gives the same result.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            min_samples_split=20,
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
    }


def train_and_compare(X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict, object, str]:
    """
    Train all 4 models, run 5-fold CV, evaluate on held-out test set.

    Returns:
        results      : dict of {model_name: metrics_dict}
        trained      : dict of {model_name: fitted model object}
        best_model   : the fitted model with the highest R² on the test set
        best_name    : name of the best model (string)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    models = get_models()
    results = {}
    trained = {}

    for name, model in models.items():
        start = time.time()

        try:
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="r2",
        n_jobs=1
    )
    cv_mean = cv_scores.mean()
except:
    cv_mean = 0
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = round(time.time() - start, 2)

        results[name] = {
            "R² (CV mean)": round(cv_scores.mean(), 4),
            "R² (CV std)": round(cv_scores.std(), 4),
            "R² (Test)": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "Train Time (s)": elapsed,
        }
        trained[name] = model

    best_name = max(results, key=lambda k: results[k]["R² (Test)"])
    best_model = trained[best_name]

    return results, trained, best_model, best_name


def save_model(model: object, encoders: dict) -> None:
    """
    Persist the best model and encoders to disk with joblib.

    WHY SAVE ENCODERS TOO?
    The encoders learned the mapping from string labels to integers
    during training (e.g., "MNC" → 2). At prediction time, we must
    apply the exact same mapping. Without saving the encoders,
    new predictions would break or produce wrong results.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)


def load_model() -> tuple[object, dict]:
    """
    Load the saved model and encoders from disk.
    Returns (None, None) if no model has been saved yet.
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders
    return None, None


def model_is_saved() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)


def get_feature_importance(model: object, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importance from tree-based models.

    Linear Regression doesn't have feature_importances_, so we skip it.
    Tree-based models (RF, XGBoost, DT) expose this attribute.

    Feature importance answers: "Which input variable did the model
    rely on most when making predictions?" This is beginner-level
    explainability and is always impressive in a portfolio.
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
        return df
    return pd.DataFrame()
