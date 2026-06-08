"""
feature_engineering.py
-----------------------
Transforms raw columns into model-ready features.

WHY FEATURE ENGINEERING?
Machine learning models only understand numbers — they cannot process
text like "Senior" or "MNC". Feature engineering converts raw data
into meaningful numeric inputs.

This is one of the highest-impact skills in a data scientist's toolkit.
A well-engineered feature can improve model accuracy more than
switching to a fancier algorithm.

Features created:
  - total_skills      : How many of the 5 skills a job requires (0-5)
  - seniority_score   : Ordinal encoding of experience_level (1/2/3)
  - is_technical_role : 1 if AI Engineer or ML Engineer, 0 otherwise
  - All categoricals label-encoded to integers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


FEATURE_COLS = [
    "company_size", "company_industry", "country", "remote_type",
    "experience_level", "education_level", "hiring_urgency",
    "years_experience", "skills_python", "skills_sql", "skills_ml",
    "skills_deep_learning", "skills_cloud",
    "total_skills", "seniority_score", "is_technical_role"
]

TARGET_COL = "salary"

CATEGORICAL_COLS = [
    "company_size", "company_industry", "country",
    "remote_type", "experience_level", "education_level", "hiring_urgency"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe.
    Returns a new dataframe — never modifies the original.

    Always return a copy so the raw data stays clean for EDA.
    """
    df = df.copy()

    df["total_skills"] = (
        df["skills_python"] + df["skills_sql"] + df["skills_ml"] +
        df["skills_deep_learning"] + df["skills_cloud"]
    )

    seniority_map = {"Entry": 1, "Mid": 2, "Senior": 3}
    df["seniority_score"] = df["experience_level"].map(seniority_map)

    technical_roles = {"AI Engineer", "Machine Learning Engineer"}
    df["is_technical_role"] = df["job_title"].apply(
        lambda x: 1 if x in technical_roles else 0
    )

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode all categorical columns.

    WHY LABEL ENCODING (not One-Hot)?
    Tree-based models (Decision Tree, Random Forest, XGBoost) handle
    label encoding well and don't need one-hot encoding. One-hot
    would create 30+ dummy columns here, adding noise with no benefit
    for these model types.

    Returns:
        df_encoded : dataframe with encoded categoricals
        encoders   : dict of {column: LabelEncoder} for inverse transforms
                     (needed to decode predictions back to readable labels)
    """
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def prepare_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full pipeline: engineer + encode + split into X and y.

    Returns:
        X        : feature matrix ready for model training
        y        : target vector (salary)
        encoders : label encoders for decoding predictions
    """
    df = engineer_features(df)
    df, encoders = encode_features(df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    return X, y, encoders


def encode_single_input(user_input: dict, encoders: dict) -> pd.DataFrame:
    """
    Encode a single user input from the Streamlit form into model format.

    This mirrors the training pipeline exactly — same columns, same order.
    If training used label encoding, prediction must use the same encoding.
    Mismatch here is the #1 cause of "model predicts garbage" bugs.

    Args:
        user_input : dict of raw values from the Streamlit form
        encoders   : the same encoders used during training

    Returns:
        A single-row DataFrame ready for model.predict()
    """
    row = user_input.copy()

    skill_cols = ["skills_python", "skills_sql", "skills_ml",
                  "skills_deep_learning", "skills_cloud"]
    row["total_skills"] = sum(row.get(c, 0) for c in skill_cols)

    seniority_map = {"Entry": 1, "Mid": 2, "Senior": 3}
    row["seniority_score"] = seniority_map.get(row.get("experience_level", "Mid"), 2)

    technical_roles = {"AI Engineer", "Machine Learning Engineer"}
    row["is_technical_role"] = 1 if row.get("job_title", "") in technical_roles else 0

    for col in CATEGORICAL_COLS:
        if col in encoders and col in row:
            try:
                row[col] = encoders[col].transform([str(row[col])])[0]
            except ValueError:
                row[col] = 0

    input_df = pd.DataFrame([row])[FEATURE_COLS]
    return input_df
