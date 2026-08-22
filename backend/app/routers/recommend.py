"""
Recommend router — cosine-similarity and full recommendation result page.
"""
import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from app.core.config import TMDB_API_KEY, TEMPLATES_DIR
from app.services.ml_engine import recommend_movies
from app.services.theme_extractor import extract_nlp_themes
from app.utils.helpers import convert_to_list

router = APIRouter(tags=["Recommendations"])
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@pass_context
def _url_for(context: dict, name: str, /, **path_params):
    request = context["request"]
    if name == "static" and "filename" in path_params:
        path_params["path"] = path_params.pop("filename")
    return request.url_for(name, **path_params)


templates.env.globals["url_for"] = _url_for


@router.post("/similarity", response_class=PlainTextResponse)
async def similarity(name: str = Form(...)):
    """
    Return the 10 most similar movie titles for *name*, separated by ``---``.
    Returns a plain error message string if the title is unknown.
    """
    result = recommend_movies(name)
    if isinstance(result, str):
        return result
    return "---".join(result)


@router.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request:          Request,
    title:            str = Form(default=""),
    cast_ids:         str = Form(default=""),
    cast_names:       str = Form(default="[]"),
    cast_chars:       str = Form(default="[]"),
    cast_bdays:       str = Form(default="[]"),
    cast_bios:        str = Form(default="[]"),
    cast_places:      str = Form(default="[]"),
    cast_profiles:    str = Form(default="[]"),
    imdb_id:          str = Form(default=""),
    poster:           str = Form(default=""),
    genres:           str = Form(default=""),
    overview:         str = Form(default=""),
    rating:           str = Form(default=""),
    vote_count:       str = Form(default=""),
    release_date:     str = Form(default=""),
    runtime:          str = Form(default=""),
    status:           str = Form(default=""),
    rec_movies:       str = Form(default="[]"),
    rec_posters:      str = Form(default="[]"),
    analyzed_reviews: str = Form(default="{}"),
):
    """Render the full recommendation result page with cast, reviews, and NLP themes."""
    rec_movies_list    = convert_to_list(rec_movies)
    rec_posters_list   = convert_to_list(rec_posters)
    cast_names_list    = convert_to_list(cast_names)
    cast_chars_list    = convert_to_list(cast_chars)
    cast_profiles_list = convert_to_list(cast_profiles)
    cast_bdays_list    = convert_to_list(cast_bdays)
    cast_bios_list     = convert_to_list(cast_bios)
    cast_places_list   = convert_to_list(cast_places)

    cast_ids_list = (
        [c.strip().replace("'", "").replace('"', "")
         for c in cast_ids.strip("[]").split(",")]
        if cast_ids else []
    )

    for i in range(len(cast_bios_list)):
        cast_bios_list[i] = cast_bios_list[i].replace(r"\n", "\n").replace(r'\"', '"')

    n = len(rec_posters_list)
    movie_cards = {rec_posters_list[i]: rec_movies_list[i] for i in range(n)}

    def _safe(lst, idx):
        return lst[idx] if idx < len(lst) else ""

    casts = {
        cast_names_list[i]: [
            _safe(cast_ids_list, i),
            _safe(cast_chars_list, i),
            cast_profiles_list[i],
        ]
        for i in range(len(cast_profiles_list))
    }
    cast_details = {
        cast_names_list[i]: [
            _safe(cast_ids_list, i),
            cast_profiles_list[i],
            _safe(cast_bdays_list, i),
            _safe(cast_places_list, i),
            _safe(cast_bios_list, i),
        ]
        for i in range(len(cast_places_list))
    }

    try:
        movie_reviews = json.loads(analyzed_reviews)
    except Exception:
        movie_reviews = {}

    nlp_themes = extract_nlp_themes(overview, genres)

    return templates.TemplateResponse(
        request=request,
        name="recommend.html",
        context={
            "title":        title,
            "poster":       poster,
            "overview":     overview,
            "vote_average": rating,
            "vote_count":   vote_count,
            "release_date": release_date,
            "runtime":      runtime,
            "status":       status,
            "genres":       genres,
            "movie_cards":  movie_cards,
            "reviews":      movie_reviews,
            "casts":        casts,
            "cast_details": cast_details,
            "nlp_themes":   nlp_themes,
            "tmdb_api_key": TMDB_API_KEY,
        },
    )
