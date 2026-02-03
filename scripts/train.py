from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODEL_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR  # noqa: E402
from src.data import load_raw, clean_data, train_test_split_data  # noqa: E402
from src.modeling import build_models, build_preprocessor, build_pipeline, evaluate_models  # noqa: E402
from src.retention import simulate_retention  # noqa: E402
from src.explain import shap_summary_plot  # noqa: E402


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw()
    df = clean_data(raw_df)
    df.to_csv(PROCESSED_DIR / "cleaned_telco_churn.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split_data(df)

    preprocessor = build_preprocessor(df)

    models = build_models()
    pipelines = {name: build_pipeline(model, preprocessor) for name, model in models.items()}

    comparison = evaluate_models(pipelines, X_train, y_train, X_test, y_test)
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    best_model_name = comparison.iloc[0]["model"]
    best_pipeline = pipelines[best_model_name]
    best_pipeline.fit(X_train, y_train)

    model_path = MODEL_DIR / "churn_model.joblib"
    joblib.dump(best_pipeline, model_path)

    probas = best_pipeline.predict_proba(X_test)[:, 1]
    retention = simulate_retention(
        df=pd.concat([X_test, y_test], axis=1),
        churn_proba=probas,
        target_fraction=0.10,
        success_rate=0.30,
        incentive_cost=10.0,
    )

    metrics = {
        "best_model": best_model_name,
        "comparison": comparison.to_dict(orient="records"),
        "retention": retention,
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    shap_path = FIGURES_DIR / "shap_summary.png"
    shap_summary_plot(best_pipeline, X_train.sample(n=min(1000, len(X_train)), random_state=42), str(shap_path))

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved SHAP plot to {shap_path}")


if __name__ == "__main__":
    main()
