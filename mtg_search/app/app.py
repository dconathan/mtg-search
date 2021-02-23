from typing import List
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from mtg_search.app.api import search
from mtg_search.constants import HOME_HTML

app = FastAPI()


class Card(BaseModel):
    name: str
    text: str


class SearchResponse(BaseModel):
    cards: List[Card]


@app.get("/search")
def search_endpoint(q: str, top: int = 10) -> SearchResponse:
    top = search(q, top)
    cards = [Card(name=c.name, text=c.text) for c in top]
    return SearchResponse(cards=cards)


@app.get("/", response_class=HTMLResponse)
def home():
    if HOME_HTML.exists():
        with HOME_HTML.open() as f:
            body = f.read()
    else:
        body = "not found :("
    return body
