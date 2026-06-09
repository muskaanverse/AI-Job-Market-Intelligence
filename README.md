# 🤖 AI Job Market Intelligence & Salary Prediction System
> A full-stack data science portfolio project analyzing 10,345 AI job market records to predict salaries and benchmark career trajectories.

---

## 🎯 What This Project Does

This system answers three questions every data professional asks:

1. **"What should I be earning?"** — ML-powered salary prediction based on your role, skills, experience, and location
2. **"Am I above or below market?"** — Peer benchmarking against similar profiles in the dataset
3. **"Which skills would boost my salary the most?"** — Skill-impact simulator using model inference

---

## 🖥️ Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Home** | Project overview and quick stats |
| **📊 Dashboard** | 10+ interactive EDA charts with sidebar filters |
| **💰 Salary Predictor** | Live salary prediction + peer comparison + skill simulator |
| **📖 About** | Methodology, tech stack, and resume bullet points |

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| Source | AI Job Market Trends 2026 |
| Records | 10,345 |
| Columns | 19 |
| Missing Values | 0 |
| Countries | 7 (Australia, Canada, Germany, India, Singapore, UK, USA) |
| Industries | 6 (Finance, Healthcare, Technology, Education, Retail, E-commerce) |
| Roles | 6 (AI Engineer, ML Engineer, Data Scientist, Data Analyst, Data Engineer, Business Analyst) |
| Years | 2020 – 2026 |

---

## 🔬 Machine Learning Pipeline

### Feature Engineering
Three domain-specific features engineered from raw columns:

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `total_skills` | Sum of 5 binary skill columns (0–5) | Captures skill breadth |
| `seniority_score` | Ordinal encoding: Entry=1, Mid=2, Senior=3 | Preserves ordinal relationship |
| `is_technical_role` | 1 if AI/ML Engineer, 0 otherwise | Separates high-pay technical roles |

### Models Compared

| Model | Why Included |
|-------|-------------|
| Linear Regression | Baseline — simplest possible model |
| Decision Tree | Non-linear, interpretable, overfitting benchmark |
| Random Forest | Ensemble method, typically strong on tabular data |
| XGBoost | Gradient boosting, industry standard for structured data |

### Validation Strategy
- 80/20 train/test split (stratified by experience level)
- 5-fold cross-validation on training set
- Metrics: R², RMSE, MAE, Training Time

---

## 🛠️ Tech Stack

```
Python 3.11
├── streamlit       — Interactive web dashboard
├── pandas          — Data loading and manipulation
├── numpy           — Numerical operations
├── scikit-learn    — ML models and preprocessing
├── xgboost         — Gradient boosting model
├── plotly          — Interactive charts
└── joblib          — Model persistence
```

---

## 📁 Project Structure

```
streamlit-app/
├── app.py                         ← Home page + navigation config
├── requirements.txt               ← Python dependencies
├── README.md                      ← This file
│
├── pages/
│   ├── 1_Dashboard.py             ← EDA dashboard (10+ charts)
│   ├── 2_Salary_Prediction.py     ← ML prediction + benchmarking
│   └── 3_About.py                 ← Methodology + resume bullets
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             ← Load + cache dataset
│   ├── eda.py                     ← All Plotly chart functions
│   ├── feature_engineering.py     ← Feature creation + encoding
│   ├── model_training.py          ← Train, compare, save models
│   └── predict.py                 ← Single-profile salary prediction
│
├── data/
│   └── ai_job_market_2026.csv     ← Source dataset
│
├── models/
│   └── best_model.pkl             ← Saved best model (auto-generated)
│
└── .streamlit/
    └── config.toml                ← Server configuration
```

---

## 🚀 Running Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/ai-job-market-intelligence

# Navigate to the project
cd ai-job-market-intelligence/streamlit-app

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

---

## 📈 Key Findings

- **AI Engineers** and **ML Engineers** earn ~40% more than Data Analysts and Business Analysts
- **MNCs** pay an 18% premium over Startups and Medium companies
- **Senior** roles earn ~$49K more than Entry-level on average  
- **Skills demand** is nearly equal across all 5 skills (~50% of postings each)
- **Country** and **work arrangement** have minimal impact on salary in this dataset

---

## 📝 Resume Bullet Points

> **Built a full-stack AI salary prediction system** using XGBoost and Random Forest on 10,345 job market records across 7 countries and 6 AI roles, comparing 4 ML models with 5-fold cross-validation

> **Engineered 3 domain-specific features** (skill breadth score, seniority index, technical role flag) to improve model performance over a pure label-encoding baseline

> **Developed an interactive Streamlit career advisor dashboard** with live salary prediction, peer benchmarking by role and experience level, and a skill-impact simulator

> **Applied modular Python architecture** (data_loader, feature_engineering, model_training, predict modules) to build a production-ready ML pipeline with persistent model saving

---

## 🗂️ Skills Demonstrated

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning (Regression)
- Model Comparison & Cross-Validation
- Interactive Data Visualization (Plotly)
- Streamlit Dashboard Development
- Python Software Architecture
- Business Storytelling with Data

---

*Version 1.0 · AI Job Market Intelligence · Built with Python + Streamlit + Scikit-Learn + XGBoost + Plotly*
