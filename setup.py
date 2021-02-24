from setuptools import setup, find_packages
from mtg_search.version import __version__
from mtg_search.constants import (
    MODEL_CHECKPOINT_PATH,
    PACKAGE_DIR,
    HOME_HTML,
)
from mtg_search import logger
import sys

if not MODEL_CHECKPOINT_PATH.exists():
    # develop will be in argv if we do e.g. `pip install -e .`
    if "develop" not in sys.argv:
        logger.error("can't build a non-development package with no model")
        raise FileNotFoundError(MODEL_CHECKPOINT_PATH)

setup(
    name="mtg-search",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "tqdm",
        "fastapi",
        "pydantic",
        "mangum",
        "transformers",
        "torch",
        "pytorch-lightning",
    ],
    extras_require={"train": ["comet-ml", "rank-bm25"]},
    entry_points={
        "console_scripts": [
            "mtg-search=mtg_search.__main__:main",
        ],
    },
    package_data={
        "mtg_search": [
            str(MODEL_CHECKPOINT_PATH.relative_to(PACKAGE_DIR) / "*.torch"),
            str(HOME_HTML.relative_to(PACKAGE_DIR)),
        ]
    },
)
