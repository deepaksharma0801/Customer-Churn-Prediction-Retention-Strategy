from __future__ import annotations

import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def _get_feature_names(preprocessor: ColumnTransformer):
    feature_names = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            ohe = transformer
            ohe_features = ohe.get_feature_names_out(columns)
            feature_names.extend(ohe_features)
        elif name == "num":
            feature_names.extend(columns)
    return feature_names


def shap_summary_plot(pipeline, X_sample: pd.DataFrame, output_path: str):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = _get_feature_names(preprocessor)

    if isinstance(model, (RandomForestClassifier,)) or model.__class__.__name__.startswith("XGB"):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, X_transformed)
    else:
        explainer = shap.Explainer(model, X_transformed)

    shap_values = explainer(X_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        features=X_transformed,
        feature_names=feature_names,
        show=False,
        plot_type="bar",
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
