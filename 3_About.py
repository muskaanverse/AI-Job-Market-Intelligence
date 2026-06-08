"""
pages/3_About.py
----------------
The About page — your project's "executive summary" for recruiters.

WHY AN ABOUT PAGE MATTERS:
When a hiring manager opens your Streamlit app, they don't always
have time to explore every chart. The About page is your 30-second pitch:
what you built, why it matters, and what skills it demonstrates.
Think of it as the README for non-technical stakeholders.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(
    page_title="About | AI Job Market",
    page_icon="📖",
    layout="wide"
)

st.title("📖 About This Project")
st.markdown("---")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ## AI Job Market Intelligence & Salary Prediction System

    This is a **Version 1 portfolio project** that demonstrates an end-to-end
    data science workflow — from raw data exploration to a deployed, interactive
    machine learning application.

    ### 🎯 Business Problem
    Data professionals entering the AI job market often lack objective tools
    to answer: *"Am I being paid fairly for my skills, experience, and location?"*

    This system analyzes 10,345 real-world AI job postings and builds a
    machine learning model that predicts salary and benchmarks a user's
    profile against similar peers.

    ---

    ### 🔬 Methodology

    **Step 1 — Data Collection**
    Dataset: AI Job Market Trends 2026 · 10,345 records · 19 columns · Zero missing values

    **Step 2 — Exploratory Data Analysis**
    10+ interactive Plotly charts answering specific business questions about
    salary, skills demand, geography, company size, and hiring trends.

    **Step 3 — Feature Engineering**
    Three engineered features added:
    - `total_skills` — counts how many of 5 skills a job requires
    - `seniority_score` — ordinal encoding (Entry=1, Mid=2, Senior=3)
    - `is_technical_role` — binary flag for AI Engineer / ML Engineer roles

    **Step 4 — Model Training & Comparison**
    4 models trained with 5-fold cross-validation:
    """)

    st.markdown("""
    | Model | Purpose |
    |-------|---------|
    | Linear Regression | Baseline — checks if the problem is linearly separable |
    | Decision Tree | Non-linear rules, interpretable, prone to overfitting |
    | Random Forest | Ensemble of 100 trees, reduces overfitting |
    | XGBoost | Gradient boosting, industry standard for tabular data |
    """)

    st.markdown("""
    **Step 5 — Model Selection & Persistence**
    Best model selected by R² on held-out test set.
    Saved to disk with `joblib` so predictions are near-instant.

    **Step 6 — Dashboard**
    Built with Streamlit — fully interactive, filterable, and deployable.
    """)

with col2:
    st.markdown("### 🛠️ Tech Stack")

    tech = {
        "Language": "Python 3.11",
        "Dashboard": "Streamlit 1.x",
        "Data": "Pandas, NumPy",
        "Visualization": "Plotly Express",
        "ML Models": "Scikit-Learn, XGBoost",
        "Model Persistence": "Joblib",
        "Validation": "5-Fold Cross-Validation",
    }

    for k, v in tech.items():
        st.markdown(f"**{k}:** {v}")

    st.markdown("---")

    st.markdown("### 📦 Dataset")
    st.markdown("""
    **Source:** AI Job Market Trends 2026

    | Property | Value |
    |----------|-------|
    | Rows | 10,345 |
    | Columns | 19 |
    | Missing Values | 0 |
    | Countries | 7 |
    | Industries | 6 |
    | Roles | 6 |
    | Years | 2020–2026 |
    """)

    st.markdown("---")

    st.markdown("### 📈 Key Findings")
    st.markdown("""
    - **AI Engineers** & **ML Engineers** earn ~40% more than other AI roles
    - **MNCs** pay a ~18% premium over Startups and Medium companies
    - **Senior** roles earn ~$49K more than Entry-level on average
    - **Skills demand** is nearly equal across all 5 skills (~50% each)
    - **Country** and **work arrangement** have minimal salary impact
    """)

st.markdown("---")
st.markdown("### 🎓 Skills Demonstrated")

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown("""
    **Data Analysis**
    - Exploratory Data Analysis
    - Statistical summaries
    - Distribution analysis
    - Outlier detection
    - Business framing
    """)

with s2:
    st.markdown("""
    **Machine Learning**
    - Feature engineering
    - Label encoding pipeline
    - 4-model comparison
    - Cross-validation
    - Model persistence
    """)

with s3:
    st.markdown("""
    **Data Visualization**
    - 10+ Plotly charts
    - Interactive dashboards
    - Business storytelling
    - KPI cards
    - Peer benchmarking
    """)

with s4:
    st.markdown("""
    **Software Engineering**
    - Modular src/ structure
    - Separation of concerns
    - Caching strategy
    - Clean code & docstrings
    - Production-ready design
    """)

st.markdown("---")

st.markdown("### 📝 Resume Bullet Points")
st.markdown("""
Copy these for your resume:

> **Built a full-stack AI salary prediction system** using XGBoost and Random Forest on 10,345 job market records across 7 countries and 6 AI roles, comparing 4 ML models with 5-fold cross-validation

> **Engineered 3 domain-specific features** (skill breadth score, seniority index, technical role flag) to improve model performance over a pure label-encoding baseline

> **Developed an interactive Streamlit career advisor dashboard** with live salary prediction, peer benchmarking by role and experience level, and a skill-impact simulator

> **Applied modular Python architecture** (data_loader, feature_engineering, model_training, predict modules) to build a production-ready ML pipeline with persistent model saving
""")

st.markdown("---")
st.caption("AI Job Market Intelligence · Version 1.0 · Built with Python + Streamlit + Scikit-Learn + XGBoost + Plotly")
