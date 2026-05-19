import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsbombpy as sb
from tqdm import tqdm

OPEN_DATA_PATH = "../open-data/data"
SHOT_EVENT_COLUMNS = [
    'id',
    'index',
    'period',
    'timestamp',
    'minute',
    'second',
    'type.id',
    'type.name',
    'possession',
    'possession_team.id',
    'possession_team.name',
    'play_pattern.id',
    'play_pattern.name',
    'team.id',
    'team.name',
    'player.id',
    'player.name',
    'position.id',
    'position.name',
    'location',
    'duration',
    'related_events',
    'shot.statsbomb_xg',
    'shot.end_location',
    'shot.key_pass_id',
    'shot.first_time',
    'shot.technique.id',
    'shot.technique.name',
    'shot.body_part.id',
    'shot.body_part.name',
    'shot.type.id',
    'shot.type.name',
    'shot.outcome.id',
    'shot.outcome.name',
    'shot.freeze_frame'
]

def load_competitions() -> pd.DataFrame:
    with open(os.path.join(OPEN_DATA_PATH, "competitions.json"), "r", encoding="utf-8") as f:
        competitions = json.load(f)
    return pd.json_normalize(competitions)

def get_match_ids(competition_ids: list[int]=None) -> list[int]:
    if competition_ids is None:
        competition_ids = load_competitions()['competition_id'].unique()
    match_ids = []
    for competition_id in competition_ids:
        matches_dir = os.path.join(OPEN_DATA_PATH, "matches", str(competition_id))
        if not os.path.isdir(matches_dir):
            continue
        for filename in os.listdir(matches_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(matches_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for match in data:
                            if 'match_id' in match:
                                match_ids.append(match['match_id'])
                    elif isinstance(data, dict) and 'match_id' in data:
                        match_ids.append(data['match_id'])
    return match_ids

def load_match(match_id: int) -> pd.DataFrame:
    with open(os.path.join(OPEN_DATA_PATH, "matches", f"{match_id}.json"), "r", encoding="utf-8") as f:
        match_data = json.load(f)
    return pd.json_normalize(match_data)

def load_all_events(match_id: int) -> pd.DataFrame:
    with open(os.path.join(OPEN_DATA_PATH, "events", f"{match_id}.json"), "r", encoding="utf-8") as f:
        events_data = json.load(f)
    return pd.json_normalize(events_data)

def load_shot_events(match_id: int) -> pd.DataFrame:
    events = load_all_events(match_id)
    if 'type.name' not in events.columns:
        return pd.DataFrame(columns=SHOT_EVENT_COLUMNS)

    # Filter to shots first, then align to a stable schema.
    shots = events[events['type.name'] == 'Shot'].copy()
    return shots.reindex(columns=SHOT_EVENT_COLUMNS)

