from pathlib import Path

# # Project name or git repository
# GIT_REPOSITORY = "novartis-datathon-2020"
# PROJECTS_PATH = Path("/content/drive/MyDrive/Projects/")

# Target variable
TARGET = "volume"
TARGET_NORM = "volume_normalized"
INDEX_COLS = ["country", "brand"]

# Define folder structure
# CURRENT_DIR = Path.cwd()
CURRENT_DIR = Path(__file__).parents[1].resolve()

PATH_DATA = CURRENT_DIR / "data"
PATH_DATA.mkdir(parents=True, exist_ok=True)

PATH_DATA_PROCESSED = PATH_DATA / "processed"
PATH_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

PATH_DATA_LOGS = PATH_DATA / "logs"
PATH_DATA_LOGS.mkdir(parents=True, exist_ok=True)

MAPPING_DATA_FILENAME = PATH_DATA_PROCESSED / "mapping_volume.joblib"

PATH_MODELS = CURRENT_DIR / "models"
PATH_MODELS.mkdir(parents=True, exist_ok=True)
