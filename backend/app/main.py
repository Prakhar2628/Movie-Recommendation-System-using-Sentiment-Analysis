"""
CINEFLIX AI — FastAPI Application Entry Point.

Run from the `backend/` directory:
    uvicorn app.main:app --host 127.0.0.1 --port 5000 --reload
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import STATIC_DIR
from app.routers import home, recommend, sentiment, discovery

# ── App Factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CINEFLIX AI",
    description=(
        "Next-Generation Movie Recommendation Engine & Sentiment NLP Analyzer. "
        "Powered by Cosine Similarity ML, Hybrid Sentiment Analysis, and AI Semantic Theme Extraction."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Static Files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(home.router)
app.include_router(recommend.router)
app.include_router(sentiment.router)
app.include_router(discovery.router)

# ── Entry Point (dev) ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from app.core.config import HOST, PORT
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
