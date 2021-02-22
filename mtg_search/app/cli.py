import argparse

from mtg_search.app.api import logger, search


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="*")
    args = parser.parse_args()
    query = " ".join(args.query)
    logger.debug(f"query is: '{query}'")
    cards = search(query)
    print("top cards:\n")
    for card in cards:
        print(f"{card.name} - {card.text}")
        print()
