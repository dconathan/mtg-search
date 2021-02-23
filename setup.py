from setuptools import setup, find_packages
from mtg_search.__version__ import __version__
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
    install_requires=["tqdm", "fastapi", "pydantic", "mangum"],
    entry_points={
        "console_scripts": [
            "download-data=mtg_search.data.utils:download_data",
            "process-data=mtg_search.data.utils:process_data",
            "train=mtg_search.models.transformer:main",
            "index=mtg_search.app.api:create_index",
            "search=mtg_search.app.api:cli",
        ],
    },
    package_data={
        "mtg_search": [
            str(MODEL_CHECKPOINT_PATH.relative_to(PACKAGE_DIR) / "*"),
            str(HOME_HTML.relative_to(PACKAGE_DIR)),
        ]
    },
)
