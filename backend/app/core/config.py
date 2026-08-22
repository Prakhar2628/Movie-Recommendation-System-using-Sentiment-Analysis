"""
Core application settings.
All environment-specific config lives here — import `settings` everywhere.
"""
import os
from pathlib import Path

# ── Base Paths ────────────────────────────────────────────────────────────────
# backend/  (one level above this file: backend/app/core/config.py → backend/)
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

MODELS_DIR: Path = BASE_DIR / "models"
DATA_DIR: Path   = BASE_DIR / "data"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"

# ── TMDB API ──────────────────────────────────────────────────────────────────
TMDB_API_KEY: str = os.environ.get("TMDB_API_KEY", "")

# ── Model Artifact Paths ──────────────────────────────────────────────────────
NLP_MODEL_PATH: Path  = MODELS_DIR / "nlp_model.pkl"
VECTORIZER_PATH: Path = MODELS_DIR / "tranform.pkl"

# ── Dataset Path ──────────────────────────────────────────────────────────────
MAIN_DATA_PATH: Path = DATA_DIR / "main_data.csv"

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = "127.0.0.1"
PORT: int = 5000
