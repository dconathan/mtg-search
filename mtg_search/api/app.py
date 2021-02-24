from typing import List
from pathlib import Path
from functools import lru_cache
import logging
import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from mangum import Mangum

from mtg_search.api.main import search
from mtg_search.constants import HOME_HTML


logger = logging.getLogger(__name__)


app = FastAPI()


class Card(BaseModel):
    name: str
    text: str


class SearchResponse(BaseModel):
    cards: List[Card]


async def wake_up():
    logger.info("waking up model")
    search("wake up")


@app.get("/search")
async def search_endpoint(q: str, top: int = 10) -> SearchResponse:
    top = search(q, top)
    cards = [Card(name=c.name, text=c.text) for c in top]
    return SearchResponse(cards=cards)


@app.get("/", response_class=HTMLResponse)
async def home():
    asyncio.create_task(wake_up())
    if HOME_HTML.exists():
        with HOME_HTML.open() as f:
            body = f.read()
    else:
        body = "not found :("
    return body


handler = Mangum(app)
