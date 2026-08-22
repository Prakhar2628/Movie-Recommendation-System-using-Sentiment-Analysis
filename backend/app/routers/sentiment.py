"""
Sentiment router — Hybrid NLP sentiment analysis endpoint.
"""
import json
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from app.services.sentiment_engine import analyze_reviews

router = APIRouter(tags=["Sentiment"])

# ── Default review templates used when no real reviews are available ───────────
def _default_reviews(title: str) -> list[str]:
    return [
        f"An absolute masterpiece! {title} delivers outstanding performances and brilliant direction throughout.",
        f"Visually impressive and deeply engaging. The plot of {title} holds your attention from start to finish.",
        f"Solid storytelling and great character development, though a few pacing choices felt slightly drawn out.",
        f"A fun and captivating watch! Really enjoyed how {title} brought its central themes to life.",
        f"Felt a bit formulaic in places, but overall the cinematography and cast make it worthwhile.",
    ]


@router.post("/analyze_sentiment", tags=["Sentiment"])
async def analyze_sentiment(
    reviews: str = Form(default="[]"),
    title:   str = Form(default="This movie"),
    overview: str = Form(default=""),
):
    """
    Run the hybrid NLP sentiment classifier on a list of review strings.
    Returns ``{review_text: "Good"|"Bad"}`` for each review.
    """
    try:
        review_list: list[str] = json.loads(reviews)
    except Exception:
        review_list = []

    if not review_list:
        review_list = _default_reviews(title)

    result = analyze_reviews(review_list)
    return JSONResponse(result)
