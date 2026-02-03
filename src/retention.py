from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_retention(
    df: pd.DataFrame,
    churn_proba: np.ndarray,
    target_fraction: float = 0.10,
    success_rate: float = 0.30,
    incentive_cost: float = 10.0,
    monthly_revenue_col: str = "MonthlyCharges",
    churn_col: str = "Churn",
):
    """
    Simulate retention impact by targeting top risk customers.

    Assumptions:
    - Revenue loss is approximated by monthly charges for churned users.
    - A % of targeted churners can be retained (success_rate).
    - Each targeted user receives an incentive with a fixed cost.
    """
    df = df.copy().reset_index(drop=True)
    df["churn_proba"] = churn_proba

    cutoff = df["churn_proba"].quantile(1 - target_fraction)
    df["targeted"] = df["churn_proba"] >= cutoff

    baseline_loss = df.loc[df[churn_col] == 1, monthly_revenue_col].sum()
    targeted_churn_loss = df.loc[(df[churn_col] == 1) & (df["targeted"]), monthly_revenue_col].sum()

    expected_saved = targeted_churn_loss * success_rate
    incentive_spend = df["targeted"].sum() * incentive_cost
    net_loss = baseline_loss - expected_saved + incentive_spend

    reduction_pct = (expected_saved / baseline_loss) if baseline_loss > 0 else 0.0

    return {
        "baseline_loss": float(baseline_loss),
        "targeted_customers": int(df["targeted"].sum()),
        "expected_saved": float(expected_saved),
        "incentive_spend": float(incentive_spend),
        "net_loss": float(net_loss),
        "reduction_pct": float(reduction_pct),
    }
