"""
Home router — serves the main CINEFLIX AI dashboard.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from jinja2 import pass_context

from app.core.config import TMDB_API_KEY, TEMPLATES_DIR
from app.utils.helpers import get_suggestions

router = APIRouter()
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@pass_context
def _url_for(context: dict, name: str, /, **path_params):
    request = context["request"]
    if name == "static" and "filename" in path_params:
        path_params["path"] = path_params.pop("filename")
    return request.url_for(name, **path_params)


templates.env.globals["url_for"] = _url_for


@router.get("/", response_class=HTMLResponse, tags=["Home"])
@router.get("/home", response_class=HTMLResponse, tags=["Home"])
async def home(request: Request):
    """Render the main CINEFLIX AI dashboard."""
    suggestions = get_suggestions()
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"suggestions": suggestions, "tmdb_api_key": TMDB_API_KEY},
    )
