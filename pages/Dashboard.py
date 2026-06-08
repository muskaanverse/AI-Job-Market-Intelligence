"""
pages/1_Dashboard.py
---------------------
The EDA Dashboard — the analytical heart of the project.

This page is what interviewers will spend the most time on.
It demonstrates that you can:
  1. Ask the right business questions
  2. Choose the right chart for each question
  3. Present insights clearly to a non-technical audience

DESIGN PRINCIPLE: Every chart answers exactly one question.
The question is displayed as the chart title so the viewer
instantly knows what they're looking at.
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data, get_summary_stats
from src import eda

st.set_page_config(
    page_title="Dashboard | AI Job Market",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Market Intelligence Dashboard")
st.markdown("Explore salary trends, skills demand, and hiring patterns across the AI job market (2020–2026).")
st.markdown("---")

df_full = load_data()

st.sidebar.header("🔍 Filter Data")

countries = ["All"] + sorted(df_full["country"].astype(str).unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries)

roles = ["All"] + sorted(df_full["job_title"].astype(str).unique().tolist())
selected_role = st.sidebar.selectbox("Job Role", roles)

industries = ["All"] + sorted(df_full["company_industry"].astype(str).unique().tolist())
selected_industry = st.sidebar.selectbox("Industry", industries)

company_sizes = ["All"] + sorted(df_full["company_size"].astype(str).unique().tolist())
selected_size = st.sidebar.selectbox("Company Size", company_sizes)

year_min, year_max = int(df_full["job_posting_year"].min()), int(df_full["job_posting_year"].max())
year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))

df = df_full.copy()
if selected_country != "All":
    df = df[df["country"] == selected_country]
if selected_role != "All":
    df = df[df["job_title"] == selected_role]
if selected_industry != "All":
    df = df[df["company_industry"] == selected_industry]
if selected_size != "All":
    df = df[df["company_size"] == selected_size]
df = df[(df["job_posting_year"] >= year_range[0]) & (df["job_posting_year"] <= year_range[1])]

if len(df) == 0:
    st.warning("No data matches the selected filters. Please adjust the sidebar filters.")
    st.stop()

stats = get_summary_stats(df)

st.markdown("### 📌 Key Metrics")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Records", f"{stats['total_records']:,}")
k2.metric("Avg Salary", f"${stats['avg_salary']:,.0f}")
k3.metric("Median Salary", f"${stats['median_salary']:,.0f}")
k4.metric("Top Paying Role", stats["top_paying_role"])
k5.metric("High Urgency Jobs", f"{stats['pct_high_urgency']:.1f}%")

st.markdown("---")

st.markdown("### 💰 Salary Analysis")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(eda.plot_salary_distribution(df), use_container_width=True)
with col2:
    st.plotly_chart(eda.plot_salary_by_role(df), use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(eda.plot_salary_heatmap(df), use_container_width=True)
with col4:
    st.plotly_chart(eda.plot_education_salary(df), use_container_width=True)

st.markdown("---")
st.markdown("### 🌍 Geography & Company")

col5, col6 = st.columns(2)
with col5:
    st.plotly_chart(eda.plot_salary_by_country(df), use_container_width=True)
with col6:
    st.plotly_chart(eda.plot_company_size_salary(df), use_container_width=True)

col7, col8 = st.columns(2)
with col7:
    st.plotly_chart(eda.plot_industry_salary(df), use_container_width=True)
with col8:
    st.plotly_chart(eda.plot_remote_type_salary(df), use_container_width=True)

st.markdown("---")
st.markdown("### 🛠️ Skills & Hiring Trends")

col9, col10 = st.columns(2)
with col9:
    st.plotly_chart(eda.plot_skills_demand(df), use_container_width=True)
with col10:
    st.plotly_chart(eda.plot_skills_salary_impact(df), use_container_width=True)

col11, col12 = st.columns(2)
with col11:
    st.plotly_chart(eda.plot_postings_over_time(df), use_container_width=True)
with col12:
    st.plotly_chart(eda.plot_hiring_urgency(df), use_container_width=True)

st.markdown("---")
with st.expander("📋 View Raw Data Sample"):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing first 100 of {len(df):,} filtered records.")
