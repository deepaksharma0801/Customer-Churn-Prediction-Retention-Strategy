from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR
from src.data import download_data, load_raw, clean_data
from src.retention import simulate_retention


st.set_page_config(page_title="Churn Prediction & Retention", layout="wide")

st.title("Customer Churn Prediction + Retention Strategy")

model_path = MODEL_DIR / "churn_model.joblib"
metrics_path = REPORTS_DIR / "metrics.json"
processed_path = PROCESSED_DIR / "cleaned_telco_churn.csv"

if not model_path.exists():
    st.warning("Model artifacts not found. Run `python scripts/train.py` first.")
    st.stop()

model = joblib.load(model_path)

if processed_path.exists():
    df = pd.read_csv(processed_path)
else:
    df = clean_data(load_raw())

X = df.drop(columns=["Churn"])

y = df["Churn"]
probas = model.predict_proba(X)[:, 1]

col1, col2, col3 = st.columns(3)
col1.metric("Churn Rate", f"{y.mean():.1%}")
col2.metric("Customers", f"{len(df):,}")
col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

st.divider()

with st.expander("Model Performance", expanded=True):
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        st.write(f"**Best model:** {metrics['best_model']}")
        st.dataframe(pd.DataFrame(metrics["comparison"]))
    else:
        st.info("Metrics not found. Run `python scripts/train.py` to generate.")

st.divider()

st.subheader("Top Risk Customers")
ranked = df.copy()
ranked["churn_proba"] = probas
ranked = ranked.sort_values(by="churn_proba", ascending=False)

st.dataframe(ranked.head(20))

st.divider()

st.subheader("Retention ROI Simulator")

col_a, col_b, col_c = st.columns(3)
with col_a:
    target_fraction = st.slider("Target top % high risk", 5, 30, 10, 1) / 100
with col_b:
    success_rate = st.slider("Retention success rate", 5, 60, 30, 5) / 100
with col_c:
    incentive_cost = st.number_input("Incentive cost ($)", 0.0, 50.0, 10.0, 1.0)

roi = simulate_retention(
    df=df.copy(),
    churn_proba=probas,
    target_fraction=target_fraction,
    success_rate=success_rate,
    incentive_cost=incentive_cost,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Loss", f"${roi['baseline_loss']:.0f}")
col2.metric("Expected Saved", f"${roi['expected_saved']:.0f}")
col3.metric("Incentive Spend", f"${roi['incentive_spend']:.0f}")
col4.metric("Loss Reduction", f"{roi['reduction_pct']:.1%}")

st.caption("Revenue loss approximated by monthly charges for churned customers.")

st.divider()

st.subheader("Explainability (SHAP)")
shap_path = FIGURES_DIR / "shap_summary.png"
if shap_path.exists():
    st.image(str(shap_path), caption="Top drivers of churn (global)", use_column_width=True)
else:
    st.info("SHAP plot not found. Run `python scripts/train.py` to generate.")
