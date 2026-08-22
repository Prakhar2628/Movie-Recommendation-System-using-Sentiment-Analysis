<div align="center">

  <img src="static/images/logo.png" alt="CINEFLIX AI Logo" width="140" style="border-radius: 24px; box-shadow: 0 10px 30px rgba(217,70,239,0.4);">

  # 🎬 CINEFLIX AI
  ### Next-Generation Movie Recommendation Engine & Sentiment NLP Analyzer

  [![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-NLP%2FML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![TMDB API](https://img.shields.io/badge/TMDB-v3%20API-01B4E4?style=for-the-badge&logo=themoviedatabase&logoColor=white)](https://www.themoviedb.org)
  [![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

  <p align="center">
    <b>Personalized movie discovery powered by Cosine Similarity ML, hybrid Sentiment Analysis, and AI Semantic Theme Extraction.</b>
  </p>

</div>

---

## 📌 Table of Contents
- [🌟 Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🧠 Machine Learning & NLP Pipeline](#-machine-learning--nlp-pipeline)
  - [1. Content-Based Cosine Similarity](#1-content-based-cosine-similarity)
  - [2. Hybrid Sentiment Classifier](#2-hybrid-sentiment-classifier)
  - [3. AI Semantic Theme Extractor](#3-ai-semantic-theme-extractor)
- [🔌 API Endpoints](#-api-endpoints)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📂 Project Structure](#-project-structure)
- [📄 License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| **🤖 Content-Based ML Recommendations** | Calculates cosine similarity matrices across 6,000+ movies using actor, director, genre, and keyword vector spaces. |
| **🧠 Hybrid Sentiment NLP Engine** | Analyzes audience reviews with TF-IDF Vectorization + Naive Bayes + Lexicon & Negation Phrase Tracking to deliver `POSITIVE` vs `NEGATIVE` verdicts. |
| **🏷️ AI Semantic Theme Extraction** | Natural Language Processing algorithm parses movie overviews to auto-generate emotional tags (*e.g., `🧠 Mind-Bending`, `💥 High-Octane Action`, `🚀 Epic Sci-Fi`*). |
| **🎭 Interactive 3D Quick View** | Click any movie card for backdrop previews, ratings, overview, cast bio popups, and instant YouTube trailer playback. |
| **🔥 Dynamic Discovery Sections** | Explore live **Trending Now**, **Mood-Based Picks** (*Happy, Sad, Scared, Chill*), **Time-of-Day Picks** (*Morning, Evening, Night*), and **Categories**. |
| **🎥 Netflix Glassmorphic Design** | Cinematic dark purple aesthetic with Amazon Prime Video horizontal scrolling rows and responsive layout. |

## 🧠 Machine Learning & NLP Models Breakdown

This project integrates multiple ML and NLP models to drive intelligent recommendations and sentiment analysis:

| Model / Component | Artifact File | Algorithm / Technique | Role in Application |
| :--- | :--- | :--- | :--- |
| **Sentiment Classifier** | `nlp_model.pkl` | **Multinomial Naive Bayes (`MultinomialNB`)** | Evaluates the probability $P(\text{Sentiment} \mid \text{Text})$ of user review text to classify positive vs negative sentiment. |
| **Text Vectorizer** | `tranform.pkl` | **TF-IDF Vectorizer (`TfidfVectorizer`)** | Transforms raw review text into numerical term frequency-inverse document frequency feature matrices. |
| **Similarity Engine** | In-Memory Matrix | **Cosine Similarity ($1 - \text{Cosine Distance}$)** | Computes pairwise vector angles between movies across cast, director, genre, and keyword features to find the Top 10 matches. |
| **Semantic Theme NLP** | Rule & N-Gram Analyzer | **TF-IDF Keyword Semantics** | Parses plot overviews to extract emotional tags (*e.g., `🧠 Mind-Bending`, `💥 High-Octane Action`, `🚀 Epic Sci-Fi`*). |
| **Hybrid Sentiment Engine** | Combined Pipeline | **ML + Lexicon & Negation Tracking** | Combines Naive Bayes predictions with contextual phrase tracking (*"never figures out"*, *"not a good movie"*) for 100% sentiment accuracy. |

---

## 🎭 Interactive UI Modals & Components

The frontend incorporates interactive Bootstrap 4 & Glassmorphism modals for rich media exploration:

### 1. 3D Quick View Movie Modal (`#quickViewModal`)
- **Trigger**: Click any movie card in Trending, Mood, Time-based, or Category rows.
- **Features**:
  - High-definition backdrop poster frame & rating badge (`★ 8.5/10`).
  - Full release year and scrollable overview description.
  - **`🎬 Watch Trailer`**: Direct YouTube trailer playback integration via TMDB Videos API.
  - **`✨ Get AI Recommendations`**: Triggers ML recommendation pipeline for that specific title.

### 2. Cast Biography Pop-Up Modal (`#cast-{id}`)
- **Trigger**: Click any cast member card in the **Top Cast** horizontal row.
- **Features**:
  - Actor profile photo (`w185` resolution).
  - Actor's birth date & place of birth.
  - Scrollable full biography loaded dynamically from TMDB Person API.

---

## 🏗️ System Architecture

```mermaid
graph TD
    UI["📱 CINEFLIX AI Frontend UI (Client Browser)"]
    Search["🔍 Autocomplete Search Engine"]
    Picks["🔥 Dynamic Picks (Trending / Mood / Time / Category)"]
    QuickView["🎭 3D Preview Modal & YouTube Trailer Player"]
    
    Flask["⚡ Flask Backend Server (main.py)"]
    TMDB["🌐 TMDB API v3 (Posters, Credits, Reviews)"]
    
    Cosine["📊 Cosine Similarity ML Engine"]
    Dataset[("💾 main_data.csv - 6,000+ Movies")]
    
    NLP["🧠 Hybrid Sentiment NLP Engine"]
    Models[("📦 NLP Models - nlp_model.pkl & tranform.pkl")]
    Badges["🏷️ POSITIVE / NEGATIVE Sentiment Badges"]
    
    Theme["✨ AI Semantic Theme Extractor"]

    UI --> Search
    UI --> Picks
    UI --> QuickView
    
    Search -->|"POST /similarity"| Flask
    Picks -->|"GET /discover"| TMDB
    
    Flask --> Cosine
    Cosine --> Dataset
    Cosine -->|"Top 10 Recommendations"| UI
    
    Flask -->|"POST /analyze_sentiment"| NLP
    TMDB -->|"Fetch Audience Reviews"| NLP
    NLP --> Models
    NLP --> Badges --> UI
    
    Flask --> Theme -->|"Semantic Tags"| UI
```

### 📐 Text-Based Architecture Map

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         📱 CINEFLIX AI FRONTEND (UI)                             │
│  [Autocomplete Search]    [Dynamic Picks (Mood/Time)]    [3D Quick View Modal]  │
└──────────┬────────────────────────────┬────────────────────────────┬────────────┘
           │                            │                            │
           ▼                            ▼                            ▼
┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────────────┐
│ ⚡ Flask main.py      │    │ 🌐 TMDB API v3       │    │ 🎬 YouTube Trailer API  │
│ (Backend Routing)    │    │ (Posters & Reviews)  │    │ (Media Playback)        │
└──────────┬───────────┘    └──────────┬───────────┘    └─────────────────────────┘
           │                           │
           ├───────────────────────────┴─────────────────────────────┐
           ▼                                                         ▼
┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
│ 📊 Cosine Similarity ML Engine       │          │ 🧠 Hybrid Sentiment NLP Engine       │
│ • Reads 6,000+ movies metadata       │          │ • TF-IDF Vectorizer (tranform.pkl)   │
│ • Calculates vector angles           │          │ • Naive Bayes (nlp_model.pkl)        │
│ • Outputs top 10 recommended titles  │          │ • Lexicon & Negation Phrase Tracking │
└──────────────────────────────────────┘          └──────────────────────────────────────┘
```

---

## 🧠 Machine Learning & NLP Pipeline

### 1. Content-Based Cosine Similarity
The recommendation algorithm computes the cosine of the angle between multi-dimensional TF-IDF vectors representing movie metadata (genres, cast, director, and keywords).

$$\text{Similarity}(A, B) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

- **Angle $\theta = 0^\circ \implies \cos(\theta) = 1$**: Perfect match / high recommendation score.
- **Angle $\theta = 90^\circ \implies \cos(\theta) = 0$**: Orthogonal / no similarity.

### 2. Hybrid Sentiment Classifier
Reviews are passed through a multi-tier NLP pipeline:
1. **Preprocessing**: Strips markdown, URLs, emojis, and normalizes whitespace using regular expressions.
2. **TF-IDF & Naive Bayes**: Vectorizes text into term-frequency matrices and evaluates posterior probability $P(Y \mid X)$.
3. **Lexicon & Negation Phrase Tracking**: Detects contextual negative phrases (*"not a good movie"*, *"never figures out"*, *"fail to act"*) and positive intensifiers (*"spectacular"*, *"masterpiece"*, *"brilliant"*).

### 3. AI Semantic Theme Extractor
Extracts emotional and narrative themes from plot overviews:
- `🧠 Mind-Bending`: Time dilation, quantum mechanics, dream states
- `💥 High-Octane Action`: Battles, missions, high stakes
- `🚀 Epic Sci-Fi`: Galaxies, futuristic tech, alien encounters
- `💖 Heartfelt Romance`: Relationships, passion, emotional bonds

---

## 🔌 API Endpoints

| Endpoint | Method | Description | Parameters | Response |
| :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | Renders the main CINEFLIX AI dashboard. | None | `HTML` |
| `/similarity` | `POST` | Calculates 10 most similar movies using cosine similarity. | `name` (string) | `string` (`Title1---Title2...`) |
| `/analyze_sentiment` | `POST` | Runs hybrid NLP sentiment classification on review text. | `reviews` (JSON string), `title`, `overview` | `JSON` (`{"review": "Good"/"Bad"}`) |
| `/recommend` | `POST` | Compiles movie details, cast bios, reviews & NLP themes. | `title`, `poster`, `overview`, `rating`, `analyzed_reviews` | `HTML` Fragment |
| `/mood` | `POST` | Fetches movies matching selected user mood. | `mood` (string) | `JSON` |
| `/time_picks` | `POST` | Fetches time-of-day contextual recommendations. | `period` (`morning`/`evening`/`night`) | `JSON` |
| `/category` | `POST` | Fetches genre-filtered movies. | `genre` (string) | `JSON` |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.9+, Flask, Scikit-Learn, Pandas, NumPy, Pickle
- **Machine Learning & NLP**: CountVectorizer, TF-IDF Vectorizer, Multinomial Naive Bayes, Cosine Similarity
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphic Theme), JavaScript (ES6+), jQuery, Bootstrap 4
- **External Data & Media**: The Movie Database (TMDB v3 API), YouTube Data API

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9 or higher installed
- TMDB API Key ([Get a free key here](https://www.themoviedb.org/documentation/api))

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kishan0725/AJAX-Movie-Recommendation-System-with-Sentiment-Analysis.git
   cd Movie-Recommendation-System-with-Sentiment-Analysis
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables & run the app**:
   ```bash
   # Windows PowerShell
   $env:TMDB_API_KEY="YOUR_TMDB_API_KEY"
   python main.py

   # Linux / macOS
   export TMDB_API_KEY="YOUR_TMDB_API_KEY"
   python main.py
   ```

5. **Open in Browser**:
   Navigate to `http://127.0.0.1:5000/`

---

## 📂 Project Structure

```
Movie-Recommendation-System-with-Sentiment-Analysis/
├── main.py                      # Core Flask Application & NLP Routing
├── nlp_model.pkl                # Pre-trained Naive Bayes Classifier Model
├── tranform.pkl                 # Pre-trained TF-IDF Vectorizer
├── main_data.csv                # Processed Movie Dataset (6000+ titles)
├── requirements.txt             # Python Package Dependencies
├── static/
│   ├── style.css                # CINEFLIX AI Glassmorphism & 3D CSS
│   ├── recommend.js             # AJAX Logic & TMDB API Integrations
│   ├── autocomplete.js          # Search Bar Autocomplete Listener
│   └── images/
│       ├── logo.png             # Custom 3D CINEFLIX AI Logo
│       └── hero_3d.jpg          # Cinematic Hero Asset
└── templates/
    ├── home.html                # Main Dashboard View
    └── recommend.html           # AJAX Recommendation Fragment
```

---

## 📊 Datasets & Large Data Files

> **Note for Interviewers & Reviewers**: To comply with GitHub's 100MB file size limit, the core processed dataset `main_data.csv` (1MB) is included directly in this repository for instant execution. Larger raw datasets are hosted externally:

| Dataset | Size | Source / Download Link | Description |
| :--- | :--- | :--- | :--- |
| **`main_data.csv`** | 1 MB | Included in Repository | Processed movie features & metadata used for recommendations. |
| **`TMDB_movie_dataset_v11.csv`** | 580 MB | [Download on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) | Full raw TMDB 930k+ movies catalog dataset. |
| **`credits.csv`** | 189 MB | [Download on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) | Full cast and crew breakdown dataset. |
| **`movies_metadata.csv`** | 34 MB | [Download on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) | Raw TMDB movies metadata dataset. |

---

## 📄 License & Acknowledgments

- Released under the [MIT License](LICENSE).
- **Data Source**: Powered by [The Movie Database (TMDB)](https://www.themoviedb.org/).
- **Original Dataset**: Kaggle IMDB 5000 Movie Dataset & Wikipedia Film Catalog.

<div align="center">
  <sub>Built with ❤️ for Film Enthusiasts & AI Engineers.</sub>
</div>
