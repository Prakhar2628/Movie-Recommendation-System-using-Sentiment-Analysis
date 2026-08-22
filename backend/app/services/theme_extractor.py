"""
AI Semantic Theme Extractor Service.

Parses a movie's overview and genre string to auto-generate emotional/narrative
tags displayed on the recommendation result page.
"""

_THEME_MAPPING: list[tuple[list[str], str]] = [
    (["mind", "reality", "dream", "subconscious", "time", "space", "quantum"],         "🧠 Mind-Bending"),
    (["action", "fight", "battle", "war", "explosion", "mission", "hero", "agent"],    "💥 High-Octane Action"),
    (["dark", "murder", "killer", "crime", "investigation", "detective", "mystery"],   "🕵️ Dark Mystery"),
    (["love", "romance", "relationship", "heart", "couple", "passion"],                "💖 Heartfelt Romance"),
    (["laugh", "funny", "comedy", "humor", "hilarious", "friends"],                    "😂 Feel-Good Humor"),
    (["space", "alien", "future", "galaxy", "planet", "robot", "sci-fi"],              "🚀 Epic Sci-Fi"),
    (["scary", "ghost", "demon", "horror", "haunted", "nightmare", "survival"],        "😱 Intense Thrills"),
    (["family", "magic", "kingdom", "dragon", "animated", "journey"],                  "✨ Magical Adventure"),
]

_FALLBACK_THEMES: list[str] = ["🎬 Cinematic Storytelling", "🌟 Critically Acclaimed"]


def extract_nlp_themes(overview: str, genres: str = "") -> list[str]:
    """
    Return up to 4 semantic theme tags for a movie based on its *overview* and *genres*.
    Falls back to generic cinematic tags if no keywords match.
    """
    text = (overview + " " + genres).lower()
    themes: list[str] = []
    for keywords, tag in _THEME_MAPPING:
        if any(word in text for word in keywords):
            themes.append(tag)
    return (themes if themes else _FALLBACK_THEMES)[:4]
