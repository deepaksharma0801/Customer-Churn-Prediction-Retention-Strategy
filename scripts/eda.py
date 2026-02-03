from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import FIGURES_DIR  # noqa: E402
from src.data import load_raw, clean_data  # noqa: E402


def save_plot(fig, filename: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    df = clean_data(load_raw())

    # Churn rate
    fig, ax = plt.subplots(figsize=(5, 4))
    churn_rate = df["Churn"].mean()
    ax.bar(["Churn rate"], [churn_rate], color="#E4572E")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Overall Churn Rate")
    save_plot(fig, "churn_rate.png")

    # Tenure distribution by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", bins=30, ax=ax, kde=True)
    ax.set_title("Tenure Distribution by Churn")
    save_plot(fig, "tenure_by_churn.png")

    # Contract type churn
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="Contract", y="Churn", ax=ax, estimator="mean")
    ax.set_title("Churn Rate by Contract Type")
    save_plot(fig, "contract_churn.png")

    # Monthly charges by churn
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_title("Monthly Charges by Churn")
    save_plot(fig, "monthly_charges_churn.png")


if __name__ == "__main__":
    main()
