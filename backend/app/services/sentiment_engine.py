"""
Hybrid Sentiment Analysis Engine.

Combines:
  1. Pre-trained TF-IDF + Multinomial Naive Bayes (nlp_model.pkl / tranform.pkl)
  2. Positive/Negative lexicon scoring
  3. Negation phrase boosting

Returns a dict mapping each (truncated) review → "Good" | "Bad".
"""
import re
import pickle
import numpy as np

from app.core.config import NLP_MODEL_PATH, VECTORIZER_PATH

# ── Load pre-trained artifacts once at import time ────────────────────────────
try:
    _clf        = pickle.load(open(NLP_MODEL_PATH, "rb"))
    _vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
except Exception as e:
    print(f"[sentiment_engine] Warning — could not load ML models: {e}")
    _clf        = None
    _vectorizer = None

# ── Lexicon ───────────────────────────────────────────────────────────────────
_POS_WORDS: frozenset[str] = frozenset({
    "great", "good", "masterpiece", "brilliant", "solid", "fun", "impressive", "enjoyed",
    "fantastic", "amazing", "awesome", "stylish", "excellent", "powerhouse", "beautifully",
    "striking", "love", "loved", "wonderful", "engaging", "thrilling", "captivating", "favorite",
    "best", "top-notch", "stellar", "perfect", "superb", "highlight", "recommend", "recommended",
    "spectacular", "entertaining", "phenomenal", "cool", "action", "spider-man", "spiderman", "hero",
})

_NEG_WORDS: frozenset[str] = frozenset({
    "frustrating", "mess", "bad", "terrible", "horrible", "waste", "disappointing", "poor",
    "boring", "dull", "pretentious", "paper-thin", "unfocused", "formulaic", "worst", "cliché",
    "lacks", "flawed", "weak", "struggles", "overrated", "cringe", "nonsense", "unfortunately", "fail",
})

_NEG_PHRASES: list[str] = [
    "not a good", "not good", "not positive", "never figures out",
    "was not", "one dimensional", "fail to act", "waste of time",
]


def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[\*\_\`\#\~\[\]\(\)\✅\❌]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze_reviews(review_list: list[str]) -> dict[str, str]:
    """
    Classify each review in *review_list* as "Good" or "Bad".
    Returns ``{truncated_review_text: verdict}`` mapping.
    """
    result: dict[str, str] = {}

    for raw in review_list:
        if not isinstance(raw, str) or not raw.strip():
            continue

        clean = _clean(raw)
        if not clean:
            continue

        key        = clean[:280] + ("..." if len(clean) > 280 else "")
        text_lower = clean.lower()

        pos_score = sum(1 for w in _POS_WORDS if w in text_lower)
        neg_score = sum(1 for w in _NEG_WORDS if w in text_lower)
        for phrase in _NEG_PHRASES:
            if phrase in text_lower:
                neg_score += 2

        ml_pred = None
        if _clf and _vectorizer:
            try:
                vec     = _vectorizer.transform(np.array([clean]))
                ml_pred = _clf.predict(vec)[0]
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

    return result
