from setuptools import setup, find_packages
from mtg_search.__version__ import __version__

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
    ],
    entry_points={
        "console_scripts": [
            "download-data=mtg_search.data.utils:download_data",
            "process-data=mtg_search.data.utils:process_data",
            "train=mtg_search.models.roberta:main",
            "index=mtg_search.api:create_index",
            "search=mtg_search.api:cli",
        ],
    },
    package_data={
        "mtg_search": [
            "artifacts/data/index.torch",
            "artifacts/models/tokenizer/*",
            "artifacts/models/model.ckpt",
        ]
    },
)
