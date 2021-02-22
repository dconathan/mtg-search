from functools import lru_cache
from typing import List
import logging

import torch

from mtg_search.constants import MODEL_CHECKPOINT_PATH, INDEX
from mtg_search.data.classes import Cards, Card, Index
from mtg_search.models.transformer import Model


logger = logging.getLogger(__name__)


@lru_cache(1)
def load_model():
    if not MODEL_CHECKPOINT_PATH.exists():
        logger.error(f"{MODEL_CHECKPOINT_PATH} not found, need to train model first?")
        raise FileNotFoundError(MODEL_CHECKPOINT_PATH)
    return Model.load_from_checkpoint(MODEL_CHECKPOINT_PATH)


@lru_cache(1)
def load_index() -> Index:
    if not INDEX.exists():
        return create_index()
    return torch.load(INDEX)


def search(query: str, top=10) -> List[Card]:
    model = load_model()
    q = model.embed_query(query)
    index = load_index()
    scores = torch.matmul(q, torch.transpose(index.vectors, 0, 1))
    best = scores.argsort(descending=True)[:top]
    return [index.cards[i] for i in best]


def create_index() -> Index:
    cards = Cards.load()
    model = load_model()
    vectors = model.create_index([c.text for c in cards])
    index = Index(cards=cards, vectors=vectors)
    torch.save(index, INDEX)
