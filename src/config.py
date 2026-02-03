from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_URLS = [
    "https://raw.githubusercontent.com/"
    "alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/"
    "treselle-systems/customer_churn_analysis/master/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://community.watsonanalytics.com/wp-content/uploads/2015/03/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
RAW_CSV_PATH = RAW_DIR / "telco_churn.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"
