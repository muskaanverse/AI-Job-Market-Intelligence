"""
pages/2_Salary_Prediction.py
-----------------------------
The salary prediction page — the interactive ML showcase.

This page demonstrates the full ML pipeline in action:
  1. User fills out their profile in the sidebar
  2. The form encodes inputs using the same pipeline used in training
  3. The saved model makes a prediction
  4. We compare the prediction against real dataset peers
  5. We show which skills would boost salary the most

WHY THIS IS IMPRESSIVE IN A PORTFOLIO:
Most students show a notebook with a trained model.
This shows a deployed model that anyone can use.
It answers a real question people actually care about:
"Am I being paid fairly?"
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import load_data
from src.feature_engineering import prepare_X_y, FEATURE_COLS
from src.model_training import (
    train_and_compare,
    save_model,
    load_model,
    model_is_saved,
    get_feature_importance
)
from src.predict import predict_salary, get_market_context

st.set_page_config(
    page_title="Salary Predictor | AI Job Market",
    page_icon="💰",
    layout="wide"
)

st.title("💰 AI Salary Predictor")
st.markdown("Enter your profile below to get an AI-powered salary estimate with peer benchmarking.")
st.markdown("---")

df = load_data()


@st.cache_resource(show_spinner="Training models — this takes about 30 seconds on first load...")
def get_trained_model():
    """
    Train models once and cache in memory for the session.

    @st.cache_resource caches the result across ALL users/sessions.
    This means the model is trained once when the app first starts,
    then every subsequent prediction is near-instant.

    WHY cache_resource instead of cache_data?
    cache_data is for data (DataFrames, dicts).
    cache_resource is for objects that should not be serialized —
    like ML models, database connections, and API clients.
    """
    X, y, encoders = prepare_X_y(df)

    if model_is_saved():
        model, saved_encoders = load_model()
        return model, saved_encoders, None, None, X.columns.tolist()

    results, trained, best_model, best_name = train_and_compare(X, y)
    X, y, encoders = prepare_X_y(df)
    save_model(best_model, encoders)

    return best_model, encoders, results, best_name, X.columns.tolist()


model, encoders, results, best_name, feature_names = get_trained_model()


st.sidebar.header("👤 Your Profile")

job_title = st.sidebar.selectbox(
    "Job Title",
    ["AI Engineer", "Business Analyst", "Data Analyst",
     "Data Engineer", "Data Scientist", "Machine Learning Engineer"]
)

experience_level = st.sidebar.selectbox(
    "Experience Level",
    ["Entry", "Mid", "Senior"]
)

years_experience = st.sidebar.slider("Years of Experience", 0, 14, 5)

education_level = st.sidebar.selectbox(
    "Education Level",
    ["Bachelor", "Master", "PhD"]
)

country = st.sidebar.selectbox(
    "Country",
    ["Australia", "Canada", "Germany", "India", "Singapore", "UK", "USA"]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    ["Startup", "Medium", "Enterprise", "MNC"]
)

company_industry = st.sidebar.selectbox(
    "Industry",
    ["E-commerce", "Education", "Finance", "Healthcare", "Retail", "Technology"]
)

remote_type = st.sidebar.selectbox(
    "Work Arrangement",
    ["Remote", "Hybrid", "Onsite"]
)

hiring_urgency = st.sidebar.selectbox(
    "Hiring Urgency (target role)",
    ["High", "Medium", "Low"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**🛠️ Your Skills**")
skills_python = int(st.sidebar.checkbox("Python", value=True))
skills_sql = int(st.sidebar.checkbox("SQL", value=True))
skills_ml = int(st.sidebar.checkbox("Machine Learning", value=False))
skills_deep_learning = int(st.sidebar.checkbox("Deep Learning", value=False))
skills_cloud = int(st.sidebar.checkbox("Cloud", value=False))

predict_btn = st.sidebar.button("🔮 Predict My Salary", type="primary", use_container_width=True)

if predict_btn:
    user_input = {
        "job_title": job_title,
        "company_size": company_size,
        "company_industry": company_industry,
        "country": country,
        "remote_type": remote_type,
        "experience_level": experience_level,
        "years_experience": years_experience,
        "education_level": education_level,
        "skills_python": skills_python,
        "skills_sql": skills_sql,
        "skills_ml": skills_ml,
        "skills_deep_learning": skills_deep_learning,
        "skills_cloud": skills_cloud,
        "hiring_urgency": hiring_urgency,
    }

    prediction = predict_salary(model, encoders, user_input)
    context = get_market_context(df, user_input, prediction["predicted_salary"])

    predicted = prediction["predicted_salary"]
    lower = prediction["lower_bound"]
    upper = prediction["upper_bound"]

    st.markdown("## 🎯 Prediction Results")

    r1, r2, r3 = st.columns(3)
    r1.metric(
        "Predicted Annual Salary",
        f"${predicted:,.0f}",
        help="Model's point estimate for your profile"
    )
    r2.metric(
        "Salary Range",
        f"${lower:,.0f} – ${upper:,.0f}",
        help="±10% confidence band around the prediction"
    )
    if context["percentile"] is not None:
        delta_str = f"+${context['salary_delta']:,.0f}" if context["salary_delta"] >= 0 else f"-${abs(context['salary_delta']):,.0f}"
        r3.metric(
            f"Market Position: {context['market_label']}",
            f"{context['percentile']:.0f}th Percentile",
            delta=delta_str,
            help=f"Compared to {context['peer_count']:,} peers with the same role and experience level"
        )

    st.markdown("---")
    st.markdown("### 📊 Peer Comparison")

    peer_filter = (
        (df["experience_level"] == experience_level) &
        (df["job_title"] == job_title)
    )
    peers = df[peer_filter]["salary"]

    if len(peers) > 0:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=peers,
            nbinsx=30,
            name="Peer Salaries",
            marker_color="steelblue",
            opacity=0.7
        ))
        fig.add_vline(
            x=predicted,
            line_dash="dash",
            line_color="red",
            line_width=3,
            annotation_text=f"Your Prediction: ${predicted:,.0f}",
            annotation_position="top"
        )
        fig.add_vline(
            x=peers.mean(),
            line_dash="dot",
            line_color="green",
            line_width=2,
            annotation_text=f"Peer Avg: ${peers.mean():,.0f}",
            annotation_position="bottom"
        )
        fig.update_layout(
            title=f"Your Salary vs. {len(peers):,} {experience_level} {job_title}s",
            xaxis_title="Annual Salary (USD)",
            yaxis_title="Number of Jobs",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🚀 Skill Impact Analysis")
    st.markdown("*How much would adding each missing skill boost your predicted salary?*")

    skill_boosts = []
    all_skills = {
        "Python": "skills_python",
        "SQL": "skills_sql",
        "Machine Learning": "skills_ml",
        "Deep Learning": "skills_deep_learning",
        "Cloud": "skills_cloud"
    }
    for skill_label, skill_col in all_skills.items():
        boosted_input = user_input.copy()
        boosted_input[skill_col] = 1
        boosted_pred = predict_salary(model, encoders, boosted_input)
        delta = boosted_pred["predicted_salary"] - predicted
        skill_boosts.append({
            "Skill": skill_label,
            "Current": user_input[skill_col],
            "Salary Boost": delta
        })

    boost_df = pd.DataFrame(skill_boosts)
    boost_df = boost_df[boost_df["Current"] == 0].sort_values("Salary Boost", ascending=False)

    if len(boost_df) > 0:
        fig2 = px.bar(
            boost_df,
            x="Skill",
            y="Salary Boost",
            color="Salary Boost",
            color_continuous_scale="RdYlGn",
            text="Salary Boost",
            title="Potential Salary Boost by Adding Each Missing Skill"
        )
        fig2.update_traces(texttemplate="+$%{y:,.0f}", textposition="outside")
        fig2.update_layout(
            height=360,
            coloraxis_showscale=False,
            yaxis_title="Estimated Salary Increase (USD)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("✅ You already have all 5 skills — you're at maximum skill coverage!")

    st.markdown("---")
    st.markdown("### 📋 Your Profile Summary")
    profile_data = {
        "Field": ["Job Title", "Experience Level", "Years Experience",
                  "Education", "Country", "Company Size", "Industry",
                  "Work Arrangement", "Skills"],
        "Value": [
            job_title, experience_level, f"{years_experience} years",
            education_level, country, company_size, company_industry,
            remote_type,
            ", ".join([s for s, v in zip(
                ["Python", "SQL", "ML", "Deep Learning", "Cloud"],
                [skills_python, skills_sql, skills_ml, skills_deep_learning, skills_cloud]
            ) if v == 1]) or "None selected"
        ]
    }
    st.table(pd.DataFrame(profile_data))

else:
    st.info("👈 Fill in your profile in the sidebar and click **Predict My Salary** to get started.")

    st.markdown("### 📈 How the Model Works")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Training Data**
        - 10,345 AI job market records
        - 2020–2026, 7 countries, 6 roles
        - 16 input features

        **Models Compared**
        1. Linear Regression (baseline)
        2. Decision Tree
        3. Random Forest
        4. XGBoost ← *best performer*
        """)

    with col2:
        st.markdown("""
        **Feature Engineering**
        - `total_skills` — skill breadth score (0–5)
        - `seniority_score` — ordinal encoding of experience
        - `is_technical_role` — AI/ML roles flagged separately
        - All categoricals label-encoded

        **Validation**
        - 80/20 train/test split
        - 5-fold cross-validation on training set
        - Metrics: R², RMSE, MAE
        """)

    if results:
        st.markdown("---")
        st.markdown("### 🏆 Model Comparison Results")

        results_df = pd.DataFrame(results).T.reset_index()
        results_df.columns = ["Model"] + list(results_df.columns[1:])
        results_df = results_df.sort_values("R2 (Test)", ascending=False)

        st.dataframe(
            results_df.style.highlight_max(
                subset=["R2 (CV mean)", "R2 (Test)"],
                color="lightgreen"
            ).highlight_min(
                subset=["RMSE", "MAE"],
                color="lightgreen"
            ).format({
                "R2 (CV mean)": "{:.4f}",
                "R2 (CV std)": "{:.4f}",
                "R2 (Test)": "{:.4f}",
                "RMSE": "${:,.0f}",
                "MAE": "${:,.0f}",
                "Train Time (s)": "{:.2f}s"
            }),
            use_container_width=True
        )
        if best_name:
            st.success(f"✅ Best Model: **{best_name}** — selected automatically and saved to disk.")
