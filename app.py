"""
Minimal xG demo API.

Loads the saved calibrated XGBoost model and exposes one endpoint, POST /predict.
The browser sends a few raw "picks" (shot spot, goalkeeper, defenders, body part);
this server fills in sensible defaults for everything else, runs the SAME feature
pipeline used in training (src/features.py), and returns the predicted xG.

Run locally:
    pip install -r requirements.txt
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""
from __future__ import annotations

import pickle

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features import build_features, feature_set

# ---------------------------------------------------------------------------
# Load the model once at startup.
# It's a CalibratedClassifierCV (isotonic) trained on all five feature tiers.
# ---------------------------------------------------------------------------
with open("models/xgboost_isotonic.pkl", "rb") as f:
    MODEL = pickle.load(f)

# The exact column list (and order) the model was trained on.
FEATURES = feature_set("tier1", "tier2", "tier3", "tier4", "tier5")


# ---------------------------------------------------------------------------
# Request body: only the handful of inputs the simple UI collects.
# Pitch coordinates are StatsBomb units: x in [0, 120], y in [0, 80],
# goal centre at (120, 40).
# ---------------------------------------------------------------------------
class ShotInput(BaseModel):
    shot: list[float]                 # [x, y] of the shooter / ball
    gk: list[float] | None = None     # [x, y] of the goalkeeper, or None
    defenders: list[list[float]] = []  # [[x, y], ...] outfield defenders
    body_part: str = "Right Foot"     # "Right Foot" | "Left Foot" | "Head"


def build_raw_row(inp: ShotInput) -> pd.DataFrame:
    """Turn the simple UI picks into the one-row raw frame build_features expects.

    Everything the UI doesn't ask about gets a neutral default so the engineered
    feature row is complete.
    """
    # Defenders become the freeze frame; each is an opponent (teammate=False)
    # and explicitly not the goalkeeper (the GK is handled via gk_location).
    freeze_frame = [
        {"location": [x, y], "teammate": False, "position": {"name": "Center Back"}}
        for x, y in inp.defenders
    ]

    row = {
        # Tier 1 — geometry
        "location": [inp.shot],
        # Tier 2 — shot context (defaults = the most common, neutral values)
        "shot.body_part.name": [inp.body_part],
        "shot.technique.name": ["Normal"],
        "play_pattern.name": ["Regular Play"],
        "shot.type.name": ["Open Play"],
        "under_pressure": [False],
        "shot.first_time": [False],
        "shot.follows_dribble": [False],
        # Tier 3 — goalkeeper
        "gk_location": [inp.gk],
        # Tier 4 — other players
        "shot.freeze_frame": [freeze_frame],
        # Tier 5 — key pass (none in this simplified demo)
        "kp_location": [None],
        "kp_length": [0.0],
        "kp_angle": [0.0],
        "kp_cross": [False],
        "kp_through_ball": [False],
        "kp_cut_back": [False],
    }
    return pd.DataFrame(row)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="xG demo")

# Allow the static GitHub Pages site (any origin) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: ShotInput) -> dict:
    raw = build_raw_row(inp)
    feats = build_features(raw)
    proba = MODEL.predict_proba(feats[FEATURES])[:, 1]
    return {"xg": float(proba[0])}
