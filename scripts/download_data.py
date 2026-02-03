from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data import download_data  # noqa: E402

if __name__ == "__main__":
    path = download_data()
    print(f"Downloaded dataset to {path}")
