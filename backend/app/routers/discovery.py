"""
Discovery router — Mood-based, time-of-day, and genre category endpoints.
"""
from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse

from app.services.ml_engine import (
    MOOD_GENRE_MAP,
    TIME_GENRE_MAP,
    get_movies_by_genres,
)

router = APIRouter(tags=["Discovery"])


@router.post("/mood")
async def mood_picks(mood: str = Form(default="happy")):
    """Return movies that match a user's current mood."""
    mood = mood.lower()
    genres = MOOD_GENRE_MAP.get(mood, ["Drama", "Comedy"])
    movies = get_movies_by_genres(genres, count=20)
    return JSONResponse({"mood": mood, "genres": genres, "movies": movies})


@router.post("/time_picks")
async def time_picks(period: str = Form(default="evening")):
    """Return movies suited to the time of day."""
    period = period.lower()
    genres = TIME_GENRE_MAP.get(period, ["Drama"])
    movies = get_movies_by_genres(genres, count=20)
    return JSONResponse({"period": period, "genres": genres, "movies": movies})


@router.post("/category")
async def category_picks(genre: str = Form(default="Action")):
    """Return movies filtered by a specific genre."""
    movies = get_movies_by_genres([genre], count=30)
    return JSONResponse({"genre": genre, "movies": movies})
