from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
EXPERTS_DIR = ASSETS_DIR / "experts"
OUTPUT_DIR = ROOT / "output"
STREAMLIT_DIR = ROOT / ".streamlit"

BASELINES_PATH = DATA_DIR / "expert_baselines.json"
DB_PATH = DATA_DIR / "strikesync.db"
TEMP_USER_VIDEO = OUTPUT_DIR / "temp_user.mp4"
SYNCED_VIDEO = OUTPUT_DIR / "synced_smash.avi"
SIDEBAR_IMAGE = ASSETS_DIR / "sidebar.png"


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPERTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)
