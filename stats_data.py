import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsbombpy as sb
from tqdm import tqdm

OPEN_DATA_PATH = "../open-data/data"
OUTPUT_PATH = os.path.join("data", "processed", "all_shots.csv")
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
    'under_pressure',
    'related_events',
    'shot.statsbomb_xg',
    'shot.end_location',
    'shot.key_pass_id',
    'shot.first_time',
    'shot.follows_dribble',
    'shot.technique.id',
    'shot.technique.name',
    'shot.body_part.id',
    'shot.body_part.name',
    'shot.type.id',
    'shot.type.name',
    'shot.outcome.id',
    'shot.outcome.name',
    'shot.freeze_frame',
]


def load_competitions() -> pd.DataFrame:
    with open(os.path.join(OPEN_DATA_PATH, "competitions.json"), "r", encoding="utf-8") as f:
        competitions = json.load(f)
    return pd.json_normalize(competitions)


def get_match_ids(competition_ids: list[int] = None) -> list[int]:
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


def load_all_events(match_id: int) -> pd.DataFrame:
    with open(os.path.join(OPEN_DATA_PATH, "events", f"{match_id}.json"), "r", encoding="utf-8") as f:
        events_data = json.load(f)
    return pd.json_normalize(events_data)


def _extract_gk_info(freeze_frame) -> dict:
    if not isinstance(freeze_frame, list):
        return {'gk_location': None}
    for player in freeze_frame:
        if (
            isinstance(player, dict)
            and player.get('position', {}).get('name') == 'Goalkeeper'
            and player.get('teammate') is False
        ):
            return {'gk_location': player.get('location')}
    return {'gk_location': None}


def load_shot_events(match_id: int) -> pd.DataFrame:
    events = load_all_events(match_id)
    if 'type.name' not in events.columns:
        return pd.DataFrame(columns=SHOT_EVENT_COLUMNS + ['gk_location'])

    shots = events[events['type.name'] == 'Shot'].copy()
    shots = shots.reindex(columns=SHOT_EVENT_COLUMNS)

    for col in ('under_pressure', 'shot.first_time', 'shot.follows_dribble'):
        if col in shots.columns:
            shots[col] = shots[col].where(shots[col].notna(), other=False).astype(bool)
        else:
            shots[col] = False

    gk_info = shots['shot.freeze_frame'].apply(_extract_gk_info)
    shots['gk_location'] = gk_info.apply(lambda d: d['gk_location'])

    return shots


def load_all_shots(competition_ids: list[int] = None, show_progress: bool = True) -> pd.DataFrame:
    match_ids = get_match_ids(competition_ids)
    frames = []
    iterator = tqdm(match_ids) if show_progress else match_ids
    for match_id in iterator:
        frames.append(load_shot_events(match_id))
    return pd.concat(frames, ignore_index=True)


def save_all_shots(OUTPUT_PATH=OUTPUT_PATH) -> None:
    print("Loading all shot events…")
    shots = load_all_shots()

    print(f"\nTotal shots: {len(shots):,}")
    print(f"Goals: {shots['shot.outcome.name'].eq('Goal').sum():,} ({shots['shot.outcome.name'].eq('Goal').mean():.1%})")
    print(f"GK location available: {shots['gk_location'].notna().sum():,} ({shots['gk_location'].notna().mean():.1%})")

    shots.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

