from setuptools import setup, find_packages
from mtg_search.__version__ import __version__
from mtg_search.constants import (
    MODEL_CHECKPOINT_PATH,
    TOKENIZER_JSON,
    INDEX,
    PACKAGE_DIR,
    HOME_HTML,
)
from mtg_search import logger


if not MODEL_CHECKPOINT_PATH.exists():
    logger.warning(
        f"{MODEL_CHECKPOINT_PATH} not found, there won't be a model as part of this package"
    )


setup(
    name="mtg-search",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "requests",
        "tqdm",
        "transformers",
        "torch",
        "rank-bm25",
        "pytorch-lightning",
        "comet-ml",
        "fastapi",
        "pydantic",
    ],
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
            str(INDEX.relative_to(PACKAGE_DIR)),
            str(TOKENIZER_JSON.relative_to(PACKAGE_DIR)),
            str(MODEL_CHECKPOINT_PATH.relative_to(PACKAGE_DIR)),
            str(HOME_HTML.relative_to(PACKAGE_DIR)),
        ]
    },
)
