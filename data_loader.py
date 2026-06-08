"""
data_loader.py
--------------
Loads and caches the AI Job Market dataset.

WHY CACHING?
Streamlit re-runs the entire script every time a user interacts with a widget.
Without caching, the CSV would be re-read from disk on every click.
@st.cache_data tells Streamlit: "load this once, then reuse it."
Recruiters care because it shows you understand performance optimization.
"""

import pandas as pd
import streamlit as st
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ai_job_market_2026.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load the dataset and apply clean data types.

    Returns:
        df (pd.DataFrame): The cleaned dataset ready for EDA and modeling.
    """
    df = pd.read_csv(DATA_PATH)

    categorical_cols = [
        "job_title", "company_size", "company_industry",
        "country", "remote_type", "experience_level",
        "education_level", "hiring_urgency"
    ]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    binary_cols = [
        "skills_python", "skills_sql", "skills_ml",
        "skills_deep_learning", "skills_cloud"
    ]
    for col in binary_cols:
        df[col] = df[col].astype(int)

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """
    Return high-level summary stats shown on the Dashboard KPI cards.

    These are the numbers displayed at the top of the dashboard
    (e.g., "Avg Salary: $113,438"). Separating this logic from the
    UI keeps the code clean and testable.
    """
    return {
        "total_records": len(df),
        "avg_salary": df["salary"].mean(),
        "median_salary": df["salary"].median(),
        "max_salary": df["salary"].max(),
        "min_salary": df["salary"].min(),
        "unique_countries": df["country"].nunique(),
        "unique_roles": df["job_title"].nunique(),
        "year_range": f"{df['job_posting_year'].min()} – {df['job_posting_year'].max()}",
        "top_paying_role": df.groupby("job_title")["salary"].mean().idxmax(),
        "pct_high_urgency": (df["hiring_urgency"] == "High").mean() * 100,
    }
