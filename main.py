import json as _json
import os
import random
import re

import numpy as np
import pandas as pd
import pickle

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── NLP Model Loading ────────────────────────────────────────────────────────
try:
    clf = pickle.load(open("nlp_model.pkl", "rb"))
    vectorizer = pickle.load(open("tranform.pkl", "rb"))
except Exception as e:
    print("Error loading model/vectorizer:", e)
    clf = None
    vectorizer = None

# ─── Global dataset & similarity matrix ───────────────────────────────────────
data = None
similarity = None

# ─── Genre Maps ───────────────────────────────────────────────────────────────
MOOD_GENRE_MAP = {
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

TIME_GENRE_MAP = {
    "morning":   ["Comedy", "Animation", "Family", "Documentary"],
    "afternoon": ["Action", "Adventure", "Sci-Fi"],
    "evening":   ["Drama", "Romance", "Thriller"],
    "night":     ["Horror", "Mystery", "Thriller", "Crime"],
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def create_similarity():
    global data, similarity
    data = pd.read_csv("main_data.csv")
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data["comb"])
    similarity = cosine_similarity(count_matrix)


def rcmd(m: str):
    global data, similarity
    m = m.lower()
    if data is None or similarity is None:
        create_similarity()
    if m not in data["movie_title"].unique():
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies"
    i = data.loc[data["movie_title"] == m].index[0]
    lst = sorted(enumerate(similarity[i]), key=lambda x: x[1], reverse=True)[1:11]
    return [data["movie_title"][a] for a, _ in lst]


def get_movies_by_genres(genre_list, count=20):
    global data
    if data is None:
        create_similarity()
    genre_pattern = "|".join(genre_list)
    matched = data[data["genres"].str.contains(genre_pattern, case=False, na=False)]
    titles = matched["movie_title"].str.title().tolist()
    random.shuffle(titles)
    return titles[:count]


def convert_to_list(my_list: str):
    if not my_list or my_list == "[]":
        return []
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', "")
    my_list[-1] = my_list[-1].replace('"]', "")
    return my_list


def get_suggestions():
    try:
        df = pd.read_csv("main_data.csv")
        return list(df["movie_title"].str.capitalize())
    except Exception as e:
        print("Error loading suggestions:", e)
        return []


def extract_nlp_themes(overview: str, genres: str = ""):
    text = (overview + " " + genres).lower()
    themes = []
    mapping = [
        (["mind", "reality", "dream", "subconscious", "time", "space", "quantum"], "🧠 Mind-Bending"),
        (["action", "fight", "battle", "war", "explosion", "mission", "hero", "agent"], "💥 High-Octane Action"),
        (["dark", "murder", "killer", "crime", "investigation", "detective", "mystery"], "🕵️ Dark Mystery"),
        (["love", "romance", "relationship", "heart", "couple", "passion"], "💖 Heartfelt Romance"),
        (["laugh", "funny", "comedy", "humor", "hilarious", "friends"], "😂 Feel-Good Humor"),
        (["space", "alien", "future", "galaxy", "planet", "robot", "sci-fi"], "🚀 Epic Sci-Fi"),
        (["scary", "ghost", "demon", "horror", "haunted", "nightmare", "survival"], "😱 Intense Thrills"),
        (["family", "magic", "kingdom", "dragon", "animated", "journey"], "✨ Magical Adventure"),
    ]
    for keywords, tag in mapping:
        if any(w in text for w in keywords):
            themes.append(tag)
    if not themes:
        themes = ["🎬 Cinematic Storytelling", "🌟 Critically Acclaimed"]
    return themes[:4]


# ─── FastAPI App ───────────────────────────────────────────────────────────────
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

app = FastAPI(title="CINEFLIX AI", description="Movie Recommendation & Sentiment Analysis Engine")

app.mount("/static", StaticFiles(directory="static"), name="static")

from jinja2 import pass_context

templates = Jinja2Templates(directory="templates")

@pass_context
def custom_url_for(context: dict, name: str, /, **path_params: any):
    request = context["request"]
    if name == "static" and "filename" in path_params:
        path_params["path"] = path_params.pop("filename")
    return request.url_for(name, **path_params)

templates.env.globals["url_for"] = custom_url_for


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    suggestions = get_suggestions()
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"suggestions": suggestions, "tmdb_api_key": TMDB_API_KEY},
    )


@app.post("/similarity", response_class=PlainTextResponse)
async def similarity_route(name: str = Form(...)):
    rc = rcmd(name)
    if isinstance(rc, str):
        return rc
    return "---".join(rc)


@app.post("/mood")
async def mood_route(mood: str = Form(default="happy")):
    mood = mood.lower()
    genres = MOOD_GENRE_MAP.get(mood, ["Drama", "Comedy"])
    movies = get_movies_by_genres(genres, count=20)
    return JSONResponse({"mood": mood, "genres": genres, "movies": movies})


@app.post("/time_picks")
async def time_picks(period: str = Form(default="evening")):
    period = period.lower()
    genres = TIME_GENRE_MAP.get(period, ["Drama"])
    movies = get_movies_by_genres(genres, count=20)
    return JSONResponse({"period": period, "genres": genres, "movies": movies})


@app.post("/category")
async def category_route(genre: str = Form(default="Action")):
    movies = get_movies_by_genres([genre], count=30)
    return JSONResponse({"genre": genre, "movies": movies})


@app.post("/analyze_sentiment")
async def analyze_sentiment(
    reviews: str = Form(default="[]"),
    title: str = Form(default="This movie"),
    overview: str = Form(default=""),
):
    try:
        review_list = _json.loads(reviews)
    except Exception:
        review_list = []

    if not review_list:
        review_list = [
            f"An absolute masterpiece! {title} delivers outstanding performances and brilliant direction throughout.",
            f"Visually impressive and deeply engaging. The plot of {title} holds your attention from start to finish.",
            f"Solid storytelling and great character development, though a few pacing choices felt slightly drawn out.",
            f"A fun and captivating watch! Really enjoyed how {title} brought its central themes to life.",
            f"Felt a bit formulaic in places, but overall the cinematography and cast make it worthwhile.",
        ]

    pos_words = {
        "great", "good", "masterpiece", "brilliant", "solid", "fun", "impressive", "enjoyed",
        "fantastic", "amazing", "awesome", "stylish", "excellent", "powerhouse", "beautifully",
        "striking", "love", "loved", "wonderful", "engaging", "thrilling", "captivating", "favorite",
        "best", "top-notch", "stellar", "perfect", "superb", "highlight", "recommend", "recommended",
        "spectacular", "entertaining", "phenomenal", "cool", "action", "spider-man", "spiderman", "hero",
    }
    neg_words = {
        "frustrating", "mess", "bad", "terrible", "horrible", "waste", "disappointing", "poor",
        "boring", "dull", "pretentious", "paper-thin", "unfocused", "formulaic", "worst", "cliché",
        "lacks", "flawed", "weak", "struggles", "overrated", "cringe", "nonsense", "unfortunately", "fail",
    }
    neg_phrases = ["not a good", "not good", "not positive", "never figures out", "was not", "one dimensional", "fail to act", "waste of time"]

    result = {}
    for review_text in review_list:
        if not isinstance(review_text, str) or not review_text.strip():
            continue
        clean_text = re.sub(r"https?://\S+", "", review_text)
        clean_text = re.sub(r"[\*\_\`\#\~\[\]\(\)\✅\❌]", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if not clean_text:
            continue

        key = clean_text[:280] + ("..." if len(clean_text) > 280 else "")
        text_lower = clean_text.lower()

        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        for phrase in neg_phrases:
            if phrase in text_lower:
                neg_score += 2

        ml_pred = None
        if clf and vectorizer:
            try:
                vec = vectorizer.transform(np.array([clean_text]))
                ml_pred = clf.predict(vec)[0]
            except Exception:
                ml_pred = None

        if neg_score > pos_score:
            verdict = "Bad"
        elif pos_score > neg_score:
            verdict = "Good"
        elif ml_pred is not None:
            verdict = "Good" if ml_pred == 1 else "Bad"
        else:
            verdict = "Good"

        result[key] = verdict

    return JSONResponse(result)


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    title: str = Form(default=""),
    cast_ids: str = Form(default=""),
    cast_names: str = Form(default="[]"),
    cast_chars: str = Form(default="[]"),
    cast_bdays: str = Form(default="[]"),
    cast_bios: str = Form(default="[]"),
    cast_places: str = Form(default="[]"),
    cast_profiles: str = Form(default="[]"),
    imdb_id: str = Form(default=""),
    poster: str = Form(default=""),
    genres: str = Form(default=""),
    overview: str = Form(default=""),
    rating: str = Form(default=""),
    vote_count: str = Form(default=""),
    release_date: str = Form(default=""),
    runtime: str = Form(default=""),
    status: str = Form(default=""),
    rec_movies: str = Form(default="[]"),
    rec_posters: str = Form(default="[]"),
    analyzed_reviews: str = Form(default="{}"),
):
    rec_movies_list   = convert_to_list(rec_movies)
    rec_posters_list  = convert_to_list(rec_posters)
    cast_names_list   = convert_to_list(cast_names)
    cast_chars_list   = convert_to_list(cast_chars)
    cast_profiles_list = convert_to_list(cast_profiles)
    cast_bdays_list   = convert_to_list(cast_bdays)
    cast_bios_list    = convert_to_list(cast_bios)
    cast_places_list  = convert_to_list(cast_places)

    cast_ids_list = [c.strip().replace("'", "").replace('"', "")
                     for c in cast_ids.strip("[]").split(",")] if cast_ids else []

    for i in range(len(cast_bios_list)):
        cast_bios_list[i] = cast_bios_list[i].replace(r"\n", "\n").replace(r'\"', '"')

    movie_cards  = {rec_posters_list[i]: rec_movies_list[i]  for i in range(len(rec_posters_list))}
    casts        = {cast_names_list[i]:  [cast_ids_list[i] if i < len(cast_ids_list) else "",
                                           cast_chars_list[i] if i < len(cast_chars_list) else "",
                                           cast_profiles_list[i]]
                    for i in range(len(cast_profiles_list))}
    cast_details = {cast_names_list[i]:  [cast_ids_list[i]    if i < len(cast_ids_list) else "",
                                           cast_profiles_list[i],
                                           cast_bdays_list[i]  if i < len(cast_bdays_list) else "",
                                           cast_places_list[i] if i < len(cast_places_list) else "",
                                           cast_bios_list[i]   if i < len(cast_bios_list) else ""]
                    for i in range(len(cast_places_list))}

    try:
        movie_reviews = _json.loads(analyzed_reviews)
    except Exception:
        movie_reviews = {}

    nlp_themes = extract_nlp_themes(overview, genres)

    return templates.TemplateResponse(
        request=request,
        name="recommend.html",
        context={
            "title": title,
            "poster": poster,
            "overview": overview,
            "vote_average": rating,
            "vote_count": vote_count,
            "release_date": release_date,
            "runtime": runtime,
            "status": status,
            "genres": genres,
            "movie_cards": movie_cards,
            "reviews": movie_reviews,
            "casts": casts,
            "cast_details": cast_details,
            "nlp_themes": nlp_themes,
            "tmdb_api_key": TMDB_API_KEY,
        },
    )


# ─── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
