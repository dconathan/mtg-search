"""
if schemas of these classes change, data will need to be re-processed
TODO list exact steps
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pickle
import logging
from typing import List

from mtg_search.constants import PROCESSED_DATA_PICKLE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Card:
    name: str
    text: str

    @property
    def is_valid(self):
        return self.name and self.text

    @classmethod
    def from_raw(cls, raw: dict):
        return cls(name=raw.get("name"), text=raw.get("text"))


class Cards:
    @staticmethod
    def load():
        if not PROCESSED_DATA_PICKLE.exists():
            logger.info(
                f"{PROCESSED_DATA_PICKLE} does not exist, downloading/processing raw data..."
            )
            from mtg_search.data.utils import process_data

            process_data()
        with PROCESSED_DATA_PICKLE.open("rb") as f:
            return pickle.load(f)


@dataclass
class Sample:
    query: str
    context: str

    @classmethod
    def from_card(cls, card: Card):
        return cls(query=card.name, context=card.text)

    @classmethod
    def from_cards(cls, cards: List[Card]) -> List[Sample]:
        return [cls.from_card(c) for c in cards]


@dataclass
class TrainSample:
    query: str
    positive: str
    negatives: List[str]


@dataclass
class TrainBatch:
    queries: List[str] = field(default_factory=list)  # n queries
    contexts: List[str] = field(
        default_factory=list
    )  # will always be >= n, where the nth context is the "label" of the nth query

    def __len__(self):
        return len(self.queries)


@dataclass
class Index:
    cards: List[Card]
    vectors: "torch.Tensor"
