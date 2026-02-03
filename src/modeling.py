from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "XGBoost failed to import. On macOS, install OpenMP runtime with "
        "`brew install libomp`, then reinstall xgboost if needed."
    ) from exc

from .data import get_feature_types


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_cols, numeric_cols = get_feature_types(df)

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )
    return preprocessor


def build_models(seed: int = 42):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=None),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def build_pipeline(model, preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, probas),
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def evaluate_models(models, X_train, y_train, X_test, y_test):
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})
    return pd.DataFrame(rows).sort_values(by="roc_auc", ascending=False)
