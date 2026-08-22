"""
Utility helpers shared across routes.
"""
import pandas as pd
from app.core.config import MAIN_DATA_PATH


def convert_to_list(my_list: str) -> list[str]:
    """Parse the custom JSON-like stringified list format used by the JS frontend."""
    if not my_list or my_list == "[]":
        return []
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', "")
    my_list[-1] = my_list[-1].replace('"]', "")
    return my_list


def get_suggestions() -> list[str]:
    """Return all movie titles (capitalised) for the autocomplete widget."""
    try:
        df = pd.read_csv(MAIN_DATA_PATH)
        return list(df["movie_title"].str.capitalize())
    except Exception as e:
        print(f"[helpers] Error loading suggestions: {e}")
        return []
