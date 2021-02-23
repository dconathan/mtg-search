from pathlib import Path
import os

# local directories
PACKAGE_DIR = Path(__file__).parent.absolute()
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"
DATA_DIR = ARTIFACTS_DIR / "data"
MODELS_DIR = ARTIFACTS_DIR / "models"

# data
DATA_URL = "https://mtgjson.com/api/v5/AllPrintings.json"
RAW_DATA_JSON = DATA_DIR / "AllPrintings.json"
PROCESSED_DATA_PICKLE = DATA_DIR / "cards.pickle"
DATA_MODULE_PICKLE = DATA_DIR / "module.pickle"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# model
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT")
if MODEL_CHECKPOINT:
    MODEL_CHECKPOINT_PATH = MODELS_DIR / MODEL_CHECKPOINT
else:
    MODEL_CHECKPOINT_PATH = None

# app
HOME_HTML = PACKAGE_DIR / "api" / "index.html"
