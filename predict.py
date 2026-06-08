"""
predict.py
----------
Handles salary prediction for a single user profile.

WHY A SEPARATE PREDICT MODULE?
Keeping prediction logic separate from the dashboard UI is good software design.
It means you could swap the Streamlit frontend for a Flask API or a mobile app
without changing any prediction code.
Recruiters who open your src/ folder will see this and think: "this person
understands separation of concerns."
"""

import pandas as pd
import numpy as np
from src.feature_engineering import encode_single_input


def predict_salary(model, encoders: dict, user_input: dict) -> dict:
    """
    Predict salary for a single user profile.

    Args:
        model      : trained sklearn/xgboost model
        encoders   : label encoders used during training
        user_input : dict of raw field values (strings and ints)

    Returns:
        dict with:
            predicted_salary   : the model's salary estimate
            lower_bound        : -10% confidence band
            upper_bound        : +10% confidence band
    """
    X_input = encode_single_input(user_input, encoders)

    predicted = float(model.predict(X_input)[0])

    predicted = max(predicted, 30_000)

    return {
        "predicted_salary": round(predicted, 2),
        "lower_bound": round(predicted * 0.90, 2),
        "upper_bound": round(predicted * 1.10, 2),
    }


def get_market_context(df: pd.DataFrame, user_input: dict, predicted_salary: float) -> dict:
    """
    Compare the user's predicted salary against similar profiles in the dataset.

    This is what separates a simple prediction form from a career advisor tool.
    Instead of just "your salary is $X", we tell them:
    "You're in the 72nd percentile for your role in your country."

    Args:
        df               : full dataset for peer comparison
        user_input       : raw user profile dict
        predicted_salary : the model's salary estimate

    Returns:
        dict with percentile, peer avg, peer count, market label
    """
    peer_filter = (
        (df["experience_level"] == user_input.get("experience_level", "Mid")) &
        (df["job_title"] == user_input.get("job_title", "Data Scientist"))
    )
    peers = df[peer_filter]["salary"]

    if len(peers) == 0:
        return {
            "percentile": None,
            "peer_avg": None,
            "peer_count": 0,
            "market_label": "N/A",
            "salary_delta": 0
        }

    percentile = float(np.mean(peers < predicted_salary) * 100)
    peer_avg = float(peers.mean())
    salary_delta = predicted_salary - peer_avg

    if percentile >= 75:
        market_label = "Above Market"
    elif percentile >= 40:
        market_label = "At Market"
    else:
        market_label = "Below Market"

    return {
        "percentile": round(percentile, 1),
        "peer_avg": round(peer_avg, 2),
        "peer_count": len(peers),
        "market_label": market_label,
        "salary_delta": round(salary_delta, 2)
    }
