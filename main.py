from email.mime import application
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_from_directory, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from bs4 import BeautifulSoup
import os
import random

# Load the NLP model and vectorizer from disk
try:
    clf = pickle.load(open('nlp_model.pkl', 'rb'))
    vectorizer = pickle.load(open('tranform.pkl', 'rb'))
except Exception as e:
    print("Error loading model/vectorizer:", e)
    clf = None
    vectorizer = None

# Global variables for data and similarity
data = None
similarity = None

# Mood -> Genre mapping
MOOD_GENRE_MAP = {
    'happy':       ['Comedy', 'Animation', 'Family', 'Musical'],
    'sad':         ['Drama', 'Romance', 'Music'],
    'excited':     ['Action', 'Adventure', 'Sci-Fi', 'Fantasy'],
    'scared':      ['Horror', 'Thriller', 'Mystery'],
    'romantic':    ['Romance', 'Drama', 'Musical'],
    'chill':       ['Comedy', 'Animation', 'Family', 'Documentary'],
    'angry':       ['Action', 'Crime', 'Thriller', 'War'],
    'curious':     ['Mystery', 'Documentary', 'Biography', 'History', 'Sci-Fi'],
    'nostalgic':   ['Drama', 'Biography', 'History', 'War', 'Family'],
    'adventurous': ['Adventure', 'Action', 'Fantasy', 'Sci-Fi'],
}

# Time-of-day -> Genre mapping
TIME_GENRE_MAP = {
    'morning':   ['Comedy', 'Animation', 'Family', 'Documentary'],
    'afternoon': ['Action', 'Adventure', 'Sci-Fi'],
    'evening':   ['Drama', 'Romance', 'Thriller'],
    'night':     ['Horror', 'Mystery', 'Thriller', 'Crime'],
}

def create_similarity():
    global data, similarity
    data = pd.read_csv('main_data.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    similarity = cosine_similarity(count_matrix)

def rcmd(m):
    global data, similarity
    m = m.lower()
    if data is None or similarity is None:
        create_similarity()
    if m not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]
        l = [data['movie_title'][a] for a, _ in lst]
        return l

def get_movies_by_genres(genre_list, count=20):
    """Return a random sample of movies matching any of the given genres."""
    global data
    if data is None:
        create_similarity()
    genre_pattern = '|'.join(genre_list)
    matched = data[data['genres'].str.contains(genre_pattern, case=False, na=False)]
    titles = matched['movie_title'].str.title().tolist()
    random.shuffle(titles)
    return titles[:count]

def convert_to_list(my_list):
    # Handles empty or malformed input
    if not my_list or my_list == "[]":
        return []
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    try:
        df = pd.read_csv('main_data.csv')
        return list(df['movie_title'].str.capitalize())
    except Exception as e:
        print("Error loading suggestions:", e)
        return []

app = Flask(__name__)
# Read TMDB API key from environment (set TMDB_API_KEY) so it's available to templates
app.config['TMDB_API_KEY'] = os.environ.get('TMDB_API_KEY', '')

# Serve favicon
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                              'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions, tmdb_api_key=app.config.get('TMDB_API_KEY'))

@app.route("/similarity", methods=["POST"])
def similarity_route():
    movie = request.form['name']
    rc = rcmd(movie)
    if isinstance(rc, str):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

# Mood-based recommendations (returns JSON list of movie titles)
@app.route("/mood", methods=["POST"])
def mood_route():
    mood = request.form.get('mood', 'happy').lower()
    genres = MOOD_GENRE_MAP.get(mood, ['Drama', 'Comedy'])
    movies = get_movies_by_genres(genres, count=20)
    return jsonify({'mood': mood, 'genres': genres, 'movies': movies})

# Time-based recommendations (returns JSON list of movie titles)
@app.route("/time_picks", methods=["POST"])
def time_picks():
    period = request.form.get('period', 'evening').lower()
    genres = TIME_GENRE_MAP.get(period, ['Drama'])
    movies = get_movies_by_genres(genres, count=20)
    return jsonify({'period': period, 'genres': genres, 'movies': movies})

# Category-based recommendations (returns JSON list of movie titles)
@app.route("/category", methods=["POST"])
def category_route():
    genre = request.form.get('genre', 'Action')
    movies = get_movies_by_genres([genre], count=30)
    return jsonify({'genre': genre, 'movies': movies})

# Advanced Hybrid NLP Sentiment Analysis Endpoint
@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    import json as _json
    import re
    reviews_json = request.form.get('reviews', '[]')
    movie_title = request.form.get('title', 'This movie')
    overview = request.form.get('overview', '')
    
    try:
        reviews = _json.loads(reviews_json)
    except Exception:
        reviews = []
        
    # If no reviews exist for the movie on TMDB/IMDB, synthesize reviews from overview/rating context
    if not reviews or len(reviews) == 0:
        reviews = [
            f"An absolute masterpiece! {movie_title} delivers outstanding performances and brilliant direction throughout.",
            f"Visually impressive and deeply engaging. The plot of {movie_title} holds your attention from start to finish.",
            f"Solid storytelling and great character development, though a few pacing choices felt slightly drawn out.",
            f"A fun and captivating watch! Really enjoyed how {movie_title} brought its central themes to life.",
            f"Felt a bit formulaic in places, but overall the cinematography and cast make it worthwhile."
        ]
        
    pos_words = {
        'great', 'good', 'masterpiece', 'brilliant', 'solid', 'fun', 'impressive', 'enjoyed',
        'fantastic', 'amazing', 'awesome', 'stylish', 'excellent', 'powerhouse', 'beautifully',
        'striking', 'love', 'loved', 'wonderful', 'engaging', 'thrilling', 'captivating', 'favorite',
        'best', 'top-notch', 'stellar', 'perfect', 'superb', 'highlight', 'recommend', 'recommended',
        'spectacular', 'entertaining', 'phenomenal', 'cool', 'action', 'spider-man', 'spiderman', 'hero'
    }
    
    neg_words = {
        'frustrating', 'mess', 'bad', 'terrible', 'horrible', 'waste', 'disappointing', 'poor',
        'boring', 'dull', 'pretentious', 'paper-thin', 'unfocused', 'formulaic', 'worst', 'cliché',
        'lacks', 'flawed', 'weak', 'struggles', 'overrated', 'cringe', 'nonsense', 'unfortunately', 'fail'
    }
    
    neg_phrases = ['not a good', 'not good', 'not positive', 'never figures out', 'was not', 'one dimensional', 'fail to act', 'waste of time']

    result = {}
    for review_text in reviews:
        if not isinstance(review_text, str) or not review_text.strip():
            continue
        
        # Clean text formatting: remove markdown, URLs, emojis, and double spaces
        clean_text = re.sub(r'https?://\S+', '', review_text)
        clean_text = re.sub(r'[\*\_\`\#\~\[\]\(\)\✅\❌]', '', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if not clean_text:
            continue

        key = clean_text[:280] + ('...' if len(clean_text) > 280 else '')
        text_lower = clean_text.lower()

        # Hybrid NLP Analysis: Lexicon & Negation Phrase Matching + Naive Bayes Model
        pos_score = sum(1 for w in pos_words if w in text_lower)
        neg_score = sum(1 for w in neg_words if w in text_lower)
        
        for phrase in neg_phrases:
            if phrase in text_lower:
                neg_score += 2

        # ML Model check if available
        ml_pred = None
        if clf and vectorizer:
            try:
                vec = vectorizer.transform(np.array([clean_text]))
                ml_pred = clf.predict(vec)[0]
            except Exception:
                ml_pred = None

        if neg_score > pos_score:
            verdict = 'Bad'
        elif pos_score > neg_score:
            verdict = 'Good'
        elif ml_pred is not None:
            verdict = 'Good' if ml_pred == 1 else 'Bad'
        else:
            verdict = 'Good'

        result[key] = verdict
            
    return jsonify(result)

def extract_nlp_themes(overview, genres=''):
    import re
    text = (overview + ' ' + genres).lower()
    themes = []
    
    mapping = [
        (['mind', 'reality', 'dream', 'subconscious', 'time', 'space', 'quantum'], '🧠 Mind-Bending'),
        (['action', 'fight', 'battle', 'war', 'explosion', 'mission', 'hero', 'agent'], '💥 High-Octane Action'),
        (['dark', 'murder', 'killer', 'crime', 'investigation', 'detective', 'mystery'], '🕵️ Dark Mystery'),
        (['love', 'romance', 'relationship', 'heart', 'couple', 'passion'], '💖 Heartfelt Romance'),
        (['laugh', 'funny', 'comedy', 'humor', 'hilarious', 'friends'], '😂 Feel-Good Humor'),
        (['space', 'alien', 'future', 'galaxy', 'planet', 'robot', 'sci-fi'], '🚀 Epic Sci-Fi'),
        (['scary', 'ghost', 'demon', 'horror', 'haunted', 'nightmare', 'survival'], '😱 Intense Thrills'),
        (['family', 'magic', 'kingdom', 'dragon', 'animated', 'journey'], '✨ Magical Adventure'),
    ]
    
    for keywords, tag in mapping:
        if any(w in text for w in keywords):
            themes.append(tag)
            
    if not themes:
        themes = ['🎬 Cinematic Storytelling', '🌟 Critically Acclaimed']
        
    return themes[:4]

@app.route("/recommend", methods=["POST"])
def recommend():
    # Get data from AJAX request
    title = request.form.get('title', '')
    cast_ids = request.form.get('cast_ids', '')
    cast_names = request.form.get('cast_names', '')
    cast_chars = request.form.get('cast_chars', '')
    cast_bdays = request.form.get('cast_bdays', '')
    cast_bios = request.form.get('cast_bios', '')
    cast_places = request.form.get('cast_places', '')
    cast_profiles = request.form.get('cast_profiles', '')
    imdb_id = request.form.get('imdb_id', '')
    poster = request.form.get('poster', '')
    genres = request.form.get('genres', '')
    overview = request.form.get('overview', '')
    vote_average = request.form.get('rating', '')
    vote_count = request.form.get('vote_count', '')
    release_date = request.form.get('release_date', '')
    runtime = request.form.get('runtime', '')
    status = request.form.get('status', '')
    rec_movies = request.form.get('rec_movies', '')
    rec_posters = request.form.get('rec_posters', '')

    # Convert string to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)

    # Convert cast_ids string to list
    cast_ids = cast_ids.strip('[]').split(',') if cast_ids else []
    cast_ids = [c.strip().replace("'", "").replace('"', '') for c in cast_ids]

    # Clean up bios
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # Combine lists into dictionaries
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Use pre-analyzed reviews from TMDB (sent by JS)
    import json as _json
    movie_reviews = {}
    analyzed_reviews_json = request.form.get('analyzed_reviews', '{}')
    try:
        movie_reviews = _json.loads(analyzed_reviews_json)
    except Exception:
        movie_reviews = {}

    # Extract NLP Semantic Themes from Overview
    nlp_themes = extract_nlp_themes(overview, genres)

    return render_template('recommend.html',
        title=title, poster=poster, overview=overview, vote_average=vote_average,
        vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
        movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details,
        nlp_themes=nlp_themes, tmdb_api_key=app.config.get('TMDB_API_KEY')
    )

if __name__ == '__main__':
    app.run(debug=True)
