# Customer Churn Prediction + Retention Strategy

End-to-end, industry-ready churn prediction project with explainable ML and ROI simulation.

## What this delivers
- Clean EDA + feature engineering prospect
- Model comparison (Logistic Regression, Random Forest, XGBoost)
- SHAP explainability for business-ready drivers
- Retention ROI simulator (top-risk targeting)
- Streamlit dashboard for stakeholders

## Dataset
- IBM Telco Customer Churn (public)
- Sources (fallback order):
  - https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv
  - https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv
  - https://community.watsonanalytics.com/wp-content/uploads/2015/03/WA_Fn-UseC_-Telco-Customer-Churn.csv

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run pipeline
```bash
python scripts/download_data.py
python scripts/eda.py
python scripts/train.py
```

## Launch dashboard
```bash
streamlit run app.py
```

## Project structure
```
.
├── app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   └── figures/
├── scripts/
│   ├── download_data.py
│   ├── eda.py
│   └── train.py
└── src/
    ├── config.py
    ├── data.py
    ├── explain.py
    ├── modeling.py
    └── retention.py
```

## Retention ROI logic
- Baseline loss = sum of MonthlyCharges for churned customers
- Target top X% highest-risk customers
- A % of targeted churners are retained (success rate)
- Incentive cost per targeted customer is subtracted

This yields a **Loss Reduction %** you can tune in the dashboard.

## Suggested talking points for recruiters
- Built an explainable churn model with **clear drivers** (contract type, tenure, monthly charges)
- Added **ROI simulator** to translate risk into retention impact
- Built a **stakeholder-ready Streamlit app** for decisioning
