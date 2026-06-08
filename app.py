"""
app.py
------
Main entry point for the AI Job Market Intelligence Dashboard.

HOW STREAMLIT MULTI-PAGE APPS WORK:
- This file (app.py) is the home page AND the navigation config.
- Any .py file inside the pages/ folder becomes a separate page automatically.
- Streamlit reads the filename to create the sidebar navigation label.

WHY THIS STRUCTURE?
One file per page keeps the codebase modular. As the app grows,
you add a new file to pages/ — you don't bloat a single 1000-line script.
This mirrors how real production dashboards are organized.
"""

import streamlit as st

st.set_page_config(
    page_title="AI Job Market Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 AI Job Market Intelligence & Salary Prediction System")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Welcome to the AI Career Intelligence Platform

    This tool analyzes **10,345 AI job market records** from 2020–2026
    across 7 countries, 6 industries, and 6 job roles to help you:

    - 📊 **Explore** salary trends, skills demand, and market patterns
    - 💰 **Predict** your market salary based on your profile
    - 🏆 **Benchmark** yourself against peers in your role and country

    ---

    **👈 Use the sidebar to navigate between pages.**
    """)

with col2:
    st.markdown("### Quick Stats")
    st.metric("Total Job Records", "10,345")
    st.metric("Countries Covered", "7")
    st.metric("Years of Data", "2020 – 2026")
    st.metric("AI Roles Analyzed", "6")

st.markdown("---")

c1, c2, c3 = st.columns(3)

with c1:
    st.info("**📊 Dashboard**\n\nInteractive EDA with 10+ charts covering salary, skills, industry, and hiring trends.")

with c2:
    st.success("**💰 Salary Predictor**\n\nEnter your profile and get an instant AI-powered salary estimate with peer benchmarking.")

with c3:
    st.warning("**📖 About**\n\nProject methodology, model comparison results, and full tech stack details.")

st.markdown("---")
st.caption("Built with Python · Streamlit · Scikit-Learn · XGBoost · Plotly · Pandas")
