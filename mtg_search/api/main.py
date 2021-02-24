from functools import lru_cache
from typing import List
import logging

from mtg_search.constants import MODEL_CHECKPOINT_PATH
from mtg_search.data.classes import Cards, Card, Index


logger = logging.getLogger(__name__)


@lru_cache(1)
def load_q_encoder():
    from mtg_search.models.transformer import Model

    logger.info(f"loading q_encoder from {MODEL_CHECKPOINT_PATH}")

    return Model.from_checkpoint_dir(MODEL_CHECKPOINT_PATH, q_only=True)


@lru_cache(1)
def load_c_encoder():
    from mtg_search.models.transformer import Model

    logger.info(f"loading c_encoder from {MODEL_CHECKPOINT_PATH}")

    return Model.from_checkpoint_dir(MODEL_CHECKPOINT_PATH, c_only=True)


@lru_cache(1)
def load_index() -> Index:
    import torch

    logger.info(f"loading index from {MODEL_CHECKPOINT_PATH}")

    index_path = MODEL_CHECKPOINT_PATH / "index.torch"
    if not index_path.exists():
        return create_index()
    return torch.load(index_path)


def search(query: str, top=10) -> List[Card]:
    import torch

    model = load_q_encoder()
    q = model.embed_query(query)
    index = load_index()
    scores = torch.matmul(q, torch.transpose(index.vectors, 0, 1))
    best = scores.argsort(descending=True)[:top]
    return [index.cards[i] for i in best]


def create_index() -> Index:
    import torch

    model = load_c_encoder()
    cards = Cards.load()
    vectors = model.create_index([c.text for c in cards])
    index = Index(cards=cards, vectors=vectors)
    index_path = MODEL_CHECKPOINT_PATH / "index.torch"
    torch.save(index, index_path)
    return index
