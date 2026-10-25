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
├── assets/experts/             # Expert reference videos
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

Reference smash clips ship in `assets/experts/`. Left-handed clips include `left` in the filename; right-handed clips include `right`.

You can add more clips to that folder. If none are present, scoring still falls back to `data/expert_baselines.json`.

```bash
python scripts/generate_baselines.py
```

## Database

No pre-built database is required. On first launch the app creates `data/strikesync.db` with empty `users` and `history` tables. Sign up to create an account, then run an analysis to start history. That file is gitignored so local accounts are not published.

## License

MIT. See [LICENSE](LICENSE).
