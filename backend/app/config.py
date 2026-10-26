"""Settings. All keys live in the single .env at the repo root.

Values are read here once and never printed or logged.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

META_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1"
SPARK_MODEL = os.getenv("GAUGE_SPARK_MODEL", "muse-spark-1.3-contributor")
DATA_DIR = Path(os.getenv("GAUGE_DATA_DIR", ROOT / "data"))


def meta_api_key() -> str:
    key = os.getenv("MODEL_API_KEY")
    if not key:
        raise RuntimeError("MODEL_API_KEY is missing: add it to the .env file in the repo root.")
    return key
