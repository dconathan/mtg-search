from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from mtg_search.app.api import search

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
