"""
eda.py
------
All Exploratory Data Analysis (EDA) chart functions.

WHY PLOTLY?
Plotly creates interactive charts — users can hover, zoom, and filter.
Matplotlib makes static images. For a Streamlit dashboard, interactive
charts look far more professional and impressive in a portfolio.

Each function answers one specific business question.
This is how real data scientists frame EDA: questions first, charts second.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


COLORS = px.colors.qualitative.Set2
PRIMARY = "#2E86AB"
SECONDARY = "#E84855"
ACCENT = "#F9C74F"


def plot_salary_distribution(df: pd.DataFrame) -> go.Figure:
    """
    Q: What does the salary landscape look like overall?
    A histogram + box plot shows spread, median, and outliers at a glance.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Salary Distribution", "Salary Box Plot by Experience")
    )

    fig.add_trace(
        go.Histogram(
            x=df["salary"],
            nbinsx=40,
            marker_color=PRIMARY,
            opacity=0.8,
            name="Salary",
            hovertemplate="Salary: $%{x:,.0f}<br>Count: %{y}<extra></extra>"
        ),
        row=1, col=1
    )

    for level, color in zip(["Entry", "Mid", "Senior"], [COLORS[0], COLORS[1], COLORS[2]]):
        subset = df[df["experience_level"] == level]["salary"]
        fig.add_trace(
            go.Box(
                y=subset,
                name=level,
                marker_color=color,
                boxmean=True,
                hovertemplate="Level: " + level + "<br>Salary: $%{y:,.0f}<extra></extra>"
            ),
            row=1, col=2
        )

    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="Salary Overview",
        title_font_size=16,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_salary_by_role(df: pd.DataFrame) -> go.Figure:
    """
    Q: Which job titles pay the most?
    Sorted horizontal bar chart — easy to compare at a glance.
    """
    avg = (
        df.groupby("job_title", observed=True)["salary"]
        .mean()
        .sort_values()
        .reset_index()
    )
    avg.columns = ["Job Title", "Average Salary"]

    fig = px.bar(
        avg,
        x="Average Salary",
        y="Job Title",
        orientation="h",
        color="Average Salary",
        color_continuous_scale="Blues",
        text="Average Salary",
        labels={"Average Salary": "Avg Annual Salary (USD)"},
        title="Average Salary by Job Title"
    )
    fig.update_traces(
        texttemplate="$%{x:,.0f}",
        textposition="outside"
    )
    fig.update_layout(
        height=380,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_salary_by_country(df: pd.DataFrame) -> go.Figure:
    """
    Q: Does location affect salary?
    """
    avg = (
        df.groupby("country", observed=True)["salary"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg.columns = ["Country", "Average Salary"]

    fig = px.bar(
        avg,
        x="Country",
        y="Average Salary",
        color="Average Salary",
        color_continuous_scale="Teal",
        text="Average Salary",
        title="Average Salary by Country"
    )
    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    fig.update_layout(
        height=380,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_salary_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Q: What's the salary at the intersection of role × experience level?
    A pivot table heatmap is the clearest way to answer this multi-dimensional question.
    This is the chart that makes recruiters say "this person thinks analytically."
    """
    pivot = (
        df.groupby(["job_title", "experience_level"], observed=True)["salary"]
        .mean()
        .reset_index()
        .pivot(index="job_title", columns="experience_level", values="salary")
    )
    pivot = pivot[["Entry", "Mid", "Senior"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont_size=12,
            hovertemplate="Role: %{y}<br>Level: %{x}<br>Avg Salary: $%{z:,.0f}<extra></extra>"
        )
    )
    fig.update_layout(
        title="Salary Heatmap: Role × Experience Level",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_skills_demand(df: pd.DataFrame) -> go.Figure:
    """
    Q: Which skills are most in-demand?
    """
    skills = {
        "Python": df["skills_python"].mean() * 100,
        "SQL": df["skills_sql"].mean() * 100,
        "Machine Learning": df["skills_ml"].mean() * 100,
        "Deep Learning": df["skills_deep_learning"].mean() * 100,
        "Cloud": df["skills_cloud"].mean() * 100,
    }
    skill_df = pd.DataFrame(list(skills.items()), columns=["Skill", "Demand (%)"])
    skill_df = skill_df.sort_values("Demand (%)", ascending=True)

    fig = px.bar(
        skill_df,
        x="Demand (%)",
        y="Skill",
        orientation="h",
        color="Demand (%)",
        color_continuous_scale="Greens",
        text="Demand (%)",
        title="Skills Demand (% of Job Postings)"
    )
    fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside")
    fig.update_layout(
        height=340,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 65], showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_skills_salary_impact(df: pd.DataFrame) -> go.Figure:
    """
    Q: What salary premium does each skill command?
    """
    skill_cols = {
        "Python": "skills_python",
        "SQL": "skills_sql",
        "Machine Learning": "skills_ml",
        "Deep Learning": "skills_deep_learning",
        "Cloud": "skills_cloud"
    }

    rows = []
    for label, col in skill_cols.items():
        with_skill = df[df[col] == 1]["salary"].mean()
        without_skill = df[df[col] == 0]["salary"].mean()
        rows.append({
            "Skill": label,
            "With Skill": with_skill,
            "Without Skill": without_skill,
            "Premium": with_skill - without_skill
        })

    skill_df = pd.DataFrame(rows).sort_values("Premium", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="With Skill",
        x=skill_df["Skill"],
        y=skill_df["With Skill"],
        marker_color=PRIMARY,
        text=[f"${v:,.0f}" for v in skill_df["With Skill"]],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name="Without Skill",
        x=skill_df["Skill"],
        y=skill_df["Without Skill"],
        marker_color="lightgray",
        text=[f"${v:,.0f}" for v in skill_df["Without Skill"]],
        textposition="outside"
    ))
    fig.update_layout(
        barmode="group",
        title="Salary With vs. Without Each Skill",
        height=400,
        yaxis_title="Average Salary (USD)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_company_size_salary(df: pd.DataFrame) -> go.Figure:
    """
    Q: Does company size (Startup vs MNC) significantly change pay?
    """
    order = ["Startup", "Medium", "Enterprise", "MNC"]
    avg = (
        df.groupby("company_size", observed=True)["salary"]
        .agg(["mean", "median", "std"])
        .reindex(order)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg["company_size"],
        y=avg["mean"],
        name="Mean Salary",
        marker_color=[COLORS[i] for i in range(4)],
        error_y=dict(type="data", array=avg["std"], visible=True),
        text=[f"${v:,.0f}" for v in avg["mean"]],
        textposition="outside",
        hovertemplate="Size: %{x}<br>Mean: $%{y:,.0f}<extra></extra>"
    ))
    fig.update_layout(
        title="Salary by Company Size (Mean ± Std Dev)",
        height=380,
        yaxis_title="Average Salary (USD)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_remote_type_salary(df: pd.DataFrame) -> go.Figure:
    """
    Q: Do remote jobs pay more, less, or the same?
    """
    avg = (
        df.groupby("remote_type", observed=True)["salary"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig = px.pie(
        avg,
        values="salary",
        names="remote_type",
        title="Salary Distribution by Work Type",
        color_discrete_sequence=COLORS,
        hole=0.4
    )
    fig.update_traces(
        texttemplate="%{label}<br>$%{value:,.0f}",
        hovertemplate="%{label}: $%{value:,.0f}<extra></extra>"
    )
    fig.update_layout(
        height=380,
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_postings_over_time(df: pd.DataFrame) -> go.Figure:
    """
    Q: How has hiring volume changed from 2020 to 2026?
    """
    trend = (
        df.groupby(["job_posting_year", "job_title"], observed=True)
        .size()
        .reset_index(name="postings")
    )

    fig = px.line(
        trend,
        x="job_posting_year",
        y="postings",
        color="job_title",
        markers=True,
        title="Job Postings Trend by Role (2020–2026)",
        labels={"job_posting_year": "Year", "postings": "Number of Postings", "job_title": "Role"}
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(dtick=1, showgrid=True, gridcolor="rgba(200,200,200,0.3)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_hiring_urgency(df: pd.DataFrame) -> go.Figure:
    """
    Q: How urgent is the market for AI talent right now?
    """
    urgency = df["hiring_urgency"].value_counts().reset_index()
    urgency.columns = ["Urgency", "Count"]

    fig = px.bar(
        urgency,
        x="Urgency",
        y="Count",
        color="Urgency",
        color_discrete_map={"High": SECONDARY, "Medium": ACCENT, "Low": COLORS[2]},
        text="Count",
        title="Hiring Urgency Distribution"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_education_salary(df: pd.DataFrame) -> go.Figure:
    """
    Q: Is a PhD worth it salary-wise?
    """
    order = ["Bachelor", "Master", "PhD"]
    avg = (
        df.groupby("education_level", observed=True)["salary"]
        .agg(["mean", "median"])
        .reindex(order)
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=avg["education_level"],
        y=avg["mean"],
        name="Mean",
        marker_color=PRIMARY,
        text=[f"${v:,.0f}" for v in avg["mean"]],
        textposition="outside"
    ))
    fig.add_trace(go.Scatter(
        x=avg["education_level"],
        y=avg["median"],
        name="Median",
        mode="markers+lines",
        marker=dict(size=12, color=SECONDARY),
        line=dict(color=SECONDARY, dash="dash")
    ))
    fig.update_layout(
        title="Salary by Education Level",
        height=380,
        yaxis_title="Salary (USD)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig


def plot_industry_salary(df: pd.DataFrame) -> go.Figure:
    """
    Q: Which industry pays the most for AI talent?
    """
    avg = (
        df.groupby("company_industry", observed=True)["salary"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    avg.columns = ["Industry", "Average Salary"]

    fig = px.bar(
        avg,
        x="Industry",
        y="Average Salary",
        color="Average Salary",
        color_continuous_scale="Purples",
        text="Average Salary",
        title="Average Salary by Industry"
    )
    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
    fig.update_layout(
        height=380,
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    )
    return fig
