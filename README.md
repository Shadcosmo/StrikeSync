# StrikeSync

AI biomechanics coaching for badminton smash technique. Upload a smash clip, compare it against expert references, and get phase-by-phase feedback on the kinetic chain.

## Features

- Pose tracking with MediaPipe
- Side-by-side sync against an expert video or a JSON baseline
- Phase scores: preparation, backswing, jump, impact, follow-through
- Local accounts, profile, and training history

## Project layout

```text
StrikeSync/
├── app.py                      # Streamlit entry point
├── strikesync/                 # Application package
│   ├── biomechanics.py         # Joint angles and scoring
│   ├── database.py             # SQLite users and history
│   └── paths.py                # Project paths
├── data/                       # Baselines (DB is created at runtime)
├── assets/experts/             # Expert reference videos (local only)
├── scripts/                    # One-off data utilities
├── output/                     # Generated analysis videos
├── .streamlit/config.toml      # Theme
└── requirements.txt
```

## Setup

Python 3.10+ recommended.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Expert videos

Place your own smash reference clips in `assets/experts/`. Name left-handed clips with `left` and right-handed clips with `right` so the tracker uses the correct arm.

These files are gitignored. Do not commit broadcast or copyrighted player footage if you publish this repository.

Without local clips, the app still scores against `data/expert_baselines.json`.

```bash
python scripts/generate_baselines.py
```

## Privacy

User accounts live in a local SQLite file at `data/strikesync.db`. That file is gitignored and should not be published.

## License

MIT. See [LICENSE](LICENSE).
