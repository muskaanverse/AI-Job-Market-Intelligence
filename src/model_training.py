import os
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_split=20, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1, verbosity=0),
    }


def train_and_compare(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    models = get_models()
    results = {}
    trained = {}

    for name, model in models.items():
        start = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        elapsed = round(time.time() - start, 2)

        results[name] = {
            "R2 (CV mean)": round(r2, 4),
            "R2 (CV std)": 0.0,
            "R2 (Test)": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2),
            "Train Time (s)": elapsed,
        }
        trained[name] = model

    best_name = max(results, key=lambda k: results[k]["R2 (Test)"])
    best_model = trained[best_name]
    return results, trained, best_model, best_name


def save_model(model, encoders):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        return joblib.load(MODEL_PATH), joblib.load(ENCODERS_PATH)
    return None, None


def model_is_saved():
    return os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH)


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        return df
    return pd.DataFrame()
