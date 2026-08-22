"""
ML Engine — Content-Based Cosine Similarity Recommendation Service.
"""
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import MAIN_DATA_PATH

# ── Genre Maps ────────────────────────────────────────────────────────────────
MOOD_GENRE_MAP: dict[str, list[str]] = {
    "happy":       ["Comedy", "Animation", "Family", "Musical"],
    "sad":         ["Drama", "Romance", "Music"],
    "excited":     ["Action", "Adventure", "Sci-Fi", "Fantasy"],
    "scared":      ["Horror", "Thriller", "Mystery"],
    "romantic":    ["Romance", "Drama", "Musical"],
    "chill":       ["Comedy", "Animation", "Family", "Documentary"],
    "angry":       ["Action", "Crime", "Thriller", "War"],
    "curious":     ["Mystery", "Documentary", "Biography", "History", "Sci-Fi"],
    "nostalgic":   ["Drama", "Biography", "History", "War", "Family"],
    "adventurous": ["Adventure", "Action", "Fantasy", "Sci-Fi"],
}

TIME_GENRE_MAP: dict[str, list[str]] = {
    "morning":   ["Comedy", "Animation", "Family", "Documentary"],
    "afternoon": ["Action", "Adventure", "Sci-Fi"],
    "evening":   ["Drama", "Romance", "Thriller"],
    "night":     ["Horror", "Mystery", "Thriller", "Crime"],
}

# ── In-Memory State ───────────────────────────────────────────────────────────
_data: pd.DataFrame | None = None
_similarity: object | None = None  # numpy ndarray


def _create_similarity() -> None:
    """Load dataset and compute cosine similarity matrix (called once on demand)."""
    global _data, _similarity
    _data = pd.read_csv(MAIN_DATA_PATH)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(_data["comb"])
    _similarity = cosine_similarity(count_matrix)


def _ensure_loaded() -> None:
    if _data is None or _similarity is None:
        _create_similarity()


def recommend_movies(movie_title: str) -> list[str] | str:
    """
    Return the Top 10 most similar movie titles for *movie_title*.
    Returns an error string if the title is not in the dataset.
    """
    _ensure_loaded()
    title = movie_title.lower()
    if title not in _data["movie_title"].unique():
        return (
            "Sorry! The movie you requested is not in our database. "
            "Please check the spelling or try with some other movies"
        )
    idx = _data.loc[_data["movie_title"] == title].index[0]
    scored = sorted(enumerate(_similarity[idx]), key=lambda x: x[1], reverse=True)[1:11]
    return [_data["movie_title"][i] for i, _ in scored]


def get_movies_by_genres(genre_list: list[str], count: int = 20) -> list[str]:
    """Return up to *count* random movies matching any genre in *genre_list*."""
    _ensure_loaded()
    pattern = "|".join(genre_list)
    matched = _data[_data["genres"].str.contains(pattern, case=False, na=False)]
    titles = matched["movie_title"].str.title().tolist()
    random.shuffle(titles)
    return titles[:count]
