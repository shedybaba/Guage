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


def _int(name, default):
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


MAX_UPLOAD_MB = _int("GAUGE_MAX_UPLOAD_MB", 300)
MAX_FRAMES = _int("GAUGE_MAX_FRAMES", 15000)  # SAM's own limit for one clip
MIN_FRAMES = 20
MAX_AUDIO_MB = _int("GAUGE_MAX_AUDIO_MB", 5)  # a 30 s question is well under 1 MB
VOICE_PER_HOUR = _int("GAUGE_VOICE_PER_HOUR", 300)  # at $0.18 per hour of audio this costs almost nothing
CORS_ORIGINS = [o.strip() for o in os.getenv("GAUGE_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
