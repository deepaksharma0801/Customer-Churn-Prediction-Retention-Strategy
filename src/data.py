from __future__ import annotations

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from .config import RAW_CSV_PATH, DATA_URLS, TARGET_COL, ID_COL


def download_data(force: bool = False) -> str:
    RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV_PATH.exists() and not force:
        return str(RAW_CSV_PATH)

    last_error = None
    for url in DATA_URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            RAW_CSV_PATH.write_bytes(response.content)
            return str(RAW_CSV_PATH)
        except requests.RequestException as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Failed to download dataset from all sources. "
        "Check your internet connection or update DATA_URLS."
    ) from last_error


def load_raw() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        download_data()
    return pd.read_csv(RAW_CSV_PATH)


def _clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 6, 12, 24, 48, 60, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-60", "61-72"],
    )
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"].clip(lower=1))
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["is_long_term_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _clean_total_charges(df)
    df = add_features(df)
    df[TARGET_COL] = (df[TARGET_COL] == "Yes").astype(int)
    df = df.drop(columns=[ID_COL])
    return df


def get_feature_types(df: pd.DataFrame):
    categorical_cols = [
        col for col in df.columns
        if df[col].dtype == "object" or str(df[col].dtype) == "category"
    ]
    numeric_cols = [col for col in df.columns if col not in categorical_cols + [TARGET_COL]]
    return categorical_cols, numeric_cols


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
