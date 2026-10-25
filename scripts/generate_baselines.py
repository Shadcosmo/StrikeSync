"""Regenerate synthetic expert elbow-angle baselines."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strikesync.paths import BASELINES_PATH, DATA_DIR  # noqa: E402

expert_database = {
    "expert_average": {
        "elbow_angles": [75, 80, 85, 90, 110, 135, 165, 175, 160, 120, 80, 45, 30],
        "max_extension": 175,
        "description": "Ideal biomechanical elbow extension curve.",
    },
    "aggressive_profile": {
        "elbow_angles": [70, 75, 85, 100, 125, 150, 172, 178, 155, 110, 70, 40, 25],
        "max_extension": 178,
        "description": "High-velocity snap with deeper backswing.",
    },
}

DATA_DIR.mkdir(parents=True, exist_ok=True)
with open(BASELINES_PATH, "w") as f:
    json.dump(expert_database, f, indent=4)

print(f"Wrote {BASELINES_PATH}")
