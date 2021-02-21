import requests
import logging
import json
from typing import List
import pickle

from tqdm import tqdm

from mtg_search.constants import DATA_URL, RAW_DATA_JSON, PROCESSED_DATA_PICKLE
from mtg_search.data.classes import Card

from random import Random


rng = Random(36423)


logger = logging.getLogger(__name__)


def download_data():
    logger.info(f"downloading data from {DATA_URL} to {RAW_DATA_JSON}")

    with RAW_DATA_JSON.open("w") as f, requests.get(DATA_URL, stream=True) as response:
        for buffer in response.iter_content(512, decode_unicode=True):
            f.write(buffer)
    logger.info("done downloading data")


def process_data():

    if not RAW_DATA_JSON.exists():
        download_data()

    logger.info("reading raw json file")
    with RAW_DATA_JSON.open() as f:
        raw_data = json.load(f)

    processed_cards: List[Card] = []

    n_invalid = 0
    n_duplicates = 0
    seen_names = set()

    raw_data = raw_data["data"]
    for card_set in tqdm(raw_data.values()):
        for raw_card in card_set["cards"]:
            card = Card.from_raw(raw_card)
            if card.name in seen_names:
                n_duplicates += 1
                continue
            if not card.is_valid:
                n_invalid += 1
                continue
            processed_cards.append(card)
            seen_names.add(card.name)

    logger.info(
        f"processed {len(processed_cards)} cards, got {n_invalid} invalid cards and saw {n_duplicates} duplicates"
    )

    logger.info(f"saving to {PROCESSED_DATA_PICKLE}")
    with PROCESSED_DATA_PICKLE.open("wb") as f:
        pickle.dump(processed_cards, f)
