from __future__ import annotations
import ast
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOAL_CENTER = np.array([120.0, 40.0])
GOAL_LEFT   = np.array([120.0, 36.0])
GOAL_RIGHT  = np.array([120.0, 44.0])
GOAL_WIDTH  = 8.0


# ---------------------------------------------------------------------------
# Parsers: Turns location lists into strings and back
# ---------------------------------------------------------------------------
def parse_location(v):
    """Parse '[x, y]' string back into a list."""

    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith('['):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
    return None


def parse_freeze_frame(v):
    """Parse a freeze-frame string back into a list of dicts."""

    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith('['):
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return []


# ---------------------------------------------------------------------------
# Geometric Features (Tier 1)
# ---------------------------------------------------------------------------
def _distance(loc):
    return float(np.linalg.norm(np.array(loc) - GOAL_CENTER))


def _angle(loc):
    """Angle in degrees subtended by the goal mouth from the shot location."""

    loc = np.asarray(loc, dtype=float)
    a = np.linalg.norm(loc - GOAL_LEFT)
    b = np.linalg.norm(loc - GOAL_RIGHT)
    c = GOAL_WIDTH
    cos_a = (a * a + b * b - c * c) / (2 * a * b)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def add_tier1(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['location'] = df['location'].apply(parse_location)
    df['shot_x'] = df['location'].apply(lambda l: l[0] if l else np.nan)
    df['shot_y'] = df['location'].apply(lambda l: l[1] if l else np.nan)
    df['distance_to_goal'] = df['location'].apply(lambda l: _distance(l) if l else np.nan)
    df['angle_to_goal'] = df['location'].apply(lambda l: _angle(l)    if l else np.nan)
    return df


# ---------------------------------------------------------------------------
# Shot Context (Tier 2)
# ---------------------------------------------------------------------------
BODY_PARTS  = ['Head', 'Left Foot', 'Right Foot']
TECHNIQUES  = ['Normal', 'Volley', 'Half Volley', 'Overhead Kick', 'Lob', 'Backheel', 'Diving Header']
PLAY_PATTERNS = ['Regular Play', 'From Corner', 'From Free Kick', 'From Throw In',
                 'From Counter', 'From Goal Kick', 'From Keeper', 'From Kick Off']
SHOT_TYPES  = ['Open Play', 'Free Kick', 'Corner', 'Kick Off']


def _onehot(series: pd.Series, prefix: str, categories: list[str]) -> pd.DataFrame:
    """One-hot encodes a column with a fixed category list."""

    s = series.fillna('Other').astype(str)
    out = pd.DataFrame(index=series.index)
    known = set(categories)
    for cat in categories:
        out[f'{prefix}_{cat.lower().replace(" ", "_")}'] = (s == cat).astype(int)
    out[f'{prefix}_other'] = (~s.isin(known)).astype(int)
    return out


def add_tier2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bp = _onehot(df['shot.body_part.name'], 'bp', BODY_PARTS)
    tech = _onehot(df['shot.technique.name'], 'tech', TECHNIQUES)
    pat  = _onehot(df['play_pattern.name'], 'pat', PLAY_PATTERNS)
    stype = _onehot(df['shot.type.name'], 'stype', SHOT_TYPES)

    df = pd.concat([df, bp, tech, pat, stype], axis=1)

    for col in ('under_pressure', 'shot.first_time', 'shot.follows_dribble'):
        df[col.replace('shot.', '')] = df[col].fillna(False).astype(int) if col in df.columns else 0

    is_header = (df['shot.body_part.name'] == 'Head').astype(int)
    df['header_distance'] = is_header * df['distance_to_goal']

    return df


# ---------------------------------------------------------------------------
# Goalkeeper Features (Tier 3)
# ---------------------------------------------------------------------------
def add_tier3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['gk_location'] = df['gk_location'].apply(parse_location)

    df['gk_x'] = df['gk_location'].apply(lambda l: l[0] if l else np.nan)
    df['gk_y'] = df['gk_location'].apply(lambda l: l[1] if l else np.nan)

    df['gk_distance_to_goal'] = np.sqrt((df['gk_x'] - 120) ** 2 + (df['gk_y'] - 40) ** 2)
    df['gk_distance_to_goal_line'] = 120 - df['gk_x']
    df['gk_lateral_offset'] = (df['gk_y'] - 40).abs()
    df['gk_distance_to_shot'] = np.sqrt((df['gk_x'] - df['shot_x']) ** 2 + (df['gk_y'] - df['shot_y']) ** 2)
    df['gk_off_line'] = (df['gk_distance_to_goal_line'] > 2).astype(int)

    shot = df[['shot_x', 'shot_y']].to_numpy()
    gk = df[['gk_x', 'gk_y']].to_numpy()
    to_goal = GOAL_CENTER - shot
    to_gk = gk - shot
    dot = (to_goal * to_gk).sum(axis=1)
    nrm = np.linalg.norm(to_goal, axis=1) * np.linalg.norm(to_gk, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos_a = np.where(nrm > 0, dot / nrm, np.nan)
        df['shot_gk_angle'] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    return df


# ---------------------------------------------------------------------------
# Other Obstructing Players Features (Tier 4)
# ---------------------------------------------------------------------------
def _point_in_triangle(p, a, b, c) -> bool:
    """Sign-of-cross-product point-in-triangle test."""

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _freeze_frame_stats(shot_loc, freeze_frame) -> dict:
    """Compute aggregate freeze-frame features for one shot."""

    out = {'defenders_in_cone': 0,
           'distance_to_nearest_defender': np.nan,
           'n_defenders_within_3m': 0,
           'n_attackers_within_3m': 0,
           'n_players_in_frame': 0}
    
    if not freeze_frame:
        return out

    out['n_players_in_frame'] = len(freeze_frame)
    sx, sy = shot_loc

    def_dists = []
    for p in freeze_frame:
        loc = p.get('location')
        if not loc:
            continue
        is_teammate = p.get('teammate', False)
        is_gk = p.get('position', {}).get('name') == 'Goalkeeper'
        if is_gk:
            continue
        d = float(np.hypot(loc[0] - sx, loc[1] - sy))
        if not is_teammate:
            def_dists.append(d)
            if _point_in_triangle(loc, shot_loc, GOAL_LEFT, GOAL_RIGHT):
                out['defenders_in_cone'] += 1
            if d <= 3.0:
                out['n_defenders_within_3m'] += 1
        else:
            if d <= 3.0:
                out['n_attackers_within_3m'] += 1

    if def_dists:
        out['distance_to_nearest_defender'] = min(def_dists)
    return out


# ---------------------------------------------------------------------------
# Available goal angle

# From the shooter's vantage point, every obstructor (GK + outfield defenders
# between the shooter and the goal) covers a slice of the angular range
# subtended by the goal mouth. We compute the *union* of those slices so
# overlapping defenders don't double-count, then report:
#
#   gk_blocked_angle           — covered by GK only         (Tier 3)
#   defenders_blocked_angle    — covered by outfield only   (Tier 4)
#   total_blocked_angle        — union of both              (combined)
#   available_goal_angle       — angle_to_goal_rad − total_blocked
#   available_goal_fraction    — available / angle_to_goal_rad
# ---------------------------------------------------------------------------

GK_RADIUS_M = 1.0
DEFENDER_RADIUS_M = 0.5


def _angular_slice(shot_x, shot_y, obs_x, obs_y, radius) -> tuple[float, float] | None:
    """Angular range in radians blocked by an obstructor of given body radius from the shooter's vantage point."""

    if obs_x <= shot_x:
        return None
    dx, dy = obs_x - shot_x, obs_y - shot_y
    d = np.hypot(dx, dy)
    if d <= radius:
        return None
    centre = np.arctan2(dy, dx)

    half = np.arcsin(min(radius / d, 1.0))
    return (centre - half, centre + half)


def _union_length(intervals: list[tuple[float, float]]) -> float:
    """Total measure of the union of a list of (lo, hi) intervals."""

    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


def _available_goal_stats(shot_loc, gk_loc, freeze_frame) -> dict:
    """Computes angular obstruction of the goal mouth."""

    out = {'gk_blocked_angle': 0.0, 
           'defenders_blocked_angle': 0.0, 
           'total_blocked_angle': 0.0, 
           'available_goal_angle': 0.0,
           'available_goal_fraction': 0.0}
    
    if shot_loc is None:
        return out
    sx, sy = shot_loc

    a_left  = np.arctan2(GOAL_LEFT[1]  - sy, GOAL_LEFT[0]  - sx)
    a_right = np.arctan2(GOAL_RIGHT[1] - sy, GOAL_RIGHT[0] - sx)
    g_lo, g_hi = (a_left, a_right) if a_left < a_right else (a_right, a_left)
    goal_span = g_hi - g_lo
    if goal_span <= 0:
        return out

    def clip_to_goal(interval):
        lo, hi = interval
        lo = max(lo, g_lo); hi = min(hi, g_hi)
        return (lo, hi) if hi > lo else None

    gk_intervals = []
    if gk_loc is not None:
        sl = _angular_slice(sx, sy, gk_loc[0], gk_loc[1], GK_RADIUS_M)
        if sl is not None:
            clipped = clip_to_goal(sl)
            if clipped is not None:
                gk_intervals.append(clipped)
    out['gk_blocked_angle'] = _union_length(gk_intervals)


    def_intervals = []
    if freeze_frame:
        for p in freeze_frame:
            loc = p.get('location')
            if not loc or p.get('teammate', False):
                continue
            if p.get('position', {}).get('name') == 'Goalkeeper':
                continue
            sl = _angular_slice(sx, sy, loc[0], loc[1], DEFENDER_RADIUS_M)
            if sl is None:
                continue
            clipped = clip_to_goal(sl)
            if clipped is not None:
                def_intervals.append(clipped)
    out['defenders_blocked_angle'] = _union_length(def_intervals)

    out['total_blocked_angle']     = _union_length(gk_intervals + def_intervals)
    out['available_goal_angle']    = max(goal_span - out['total_blocked_angle'], 0.0)
    out['available_goal_fraction'] = out['available_goal_angle'] / goal_span
    return out


def add_tier4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['shot.freeze_frame'] = df['shot.freeze_frame'].apply(parse_freeze_frame)

    locs = df['location'].tolist()
    ffs  = df['shot.freeze_frame'].tolist()
    gks  = df['gk_location'].tolist()

    counts = [_freeze_frame_stats(loc, ff) for loc, ff in zip(locs, ffs)]
    angles = [_available_goal_stats(loc, gk, ff) for loc, gk, ff in zip(locs, gks, ffs)]

    ff_df  = pd.DataFrame(counts, index=df.index)
    ang_df = pd.DataFrame(angles, index=df.index)

    # Median-impute nearest-defender distance for shots with no opposition in frame
    med = ff_df['distance_to_nearest_defender'].median()
    ff_df['distance_to_nearest_defender'] = ff_df['distance_to_nearest_defender'].fillna(med)

    return pd.concat([df, ff_df, ang_df], axis=1)


# ---------------------------------------------------------------------------
# Key Passes, Pre-Shot Context (Tier 5)
# ---------------------------------------------------------------------------
def add_tier5(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['kp_location'] = df['kp_location'].apply(parse_location)
    df['has_key_pass'] = df['kp_location'].apply(lambda l: 1 if l else 0).astype(int)

    df['kp_x'] = df['kp_location'].apply(lambda l: l[0] if l else 0.0)
    df['kp_y'] = df['kp_location'].apply(lambda l: l[1] if l else 0.0)

    df['kp_length'] = pd.to_numeric(df['kp_length'], errors='coerce').fillna(0.0)
    df['kp_angle'] = pd.to_numeric(df['kp_angle'],  errors='coerce').fillna(0.0)

    for col in ('kp_cross', 'kp_through_ball', 'kp_cut_back'):
        df[col] = (df[col] == True).astype(int)

    df['kp_distance_to_goal'] = np.where(
        df['has_key_pass'] == 1,
        np.sqrt((df['kp_x'] - 120) ** 2 + (df['kp_y'] - 40) ** 2),
        0.0,
    )

    df['kp_forward_progress'] = np.where(
        df['has_key_pass'] == 1,
        df['kp_distance_to_goal'] - df['distance_to_goal'],
        0.0,
    )

    return df


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['goal'] = (df['shot.outcome.name'] == 'Goal').astype(int)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Add every engineered feature plus the goal label."""

    df = shots.copy()
    df = add_tier1(df)
    df = add_tier2(df)
    df = add_tier3(df)
    df = add_tier4(df)
    df = add_tier5(df)
    df = add_label(df)
    return df


TIER_COLUMNS: dict[str, list[str]] = {
    'tier1': [
        'distance_to_goal',
        'angle_to_goal',
    ],
    'tier2': (
        [f'bp_{c.lower().replace(" ", "_")}' for c in BODY_PARTS] + ['bp_other'] +
        [f'tech_{c.lower().replace(" ", "_")}' for c in TECHNIQUES] + ['tech_other'] +
        [f'pat_{c.lower().replace(" ", "_")}'  for c in PLAY_PATTERNS] + ['pat_other'] +
        [f'stype_{c.lower().replace(" ", "_")}' for c in SHOT_TYPES] + ['stype_other'] +
        ['under_pressure', 'first_time', 'follows_dribble', 'header_distance']
    ),
    'tier3': [
        'gk_distance_to_goal',
        'gk_distance_to_goal_line',
        'gk_lateral_offset',
        'gk_distance_to_shot',
        'gk_off_line',
        'shot_gk_angle',
        'gk_blocked_angle',
    ],
    'tier4': [
        'defenders_in_cone',
        'distance_to_nearest_defender',
        'n_defenders_within_3m',
        'n_attackers_within_3m',
        'n_players_in_frame',
        'defenders_blocked_angle',
        'available_goal_angle',
        'available_goal_fraction',
    ],
    'tier5': [
        'has_key_pass',
        'kp_x', 'kp_y',
        'kp_length', 'kp_angle',
        'kp_distance_to_goal',
        'kp_forward_progress',
        'kp_cross', 'kp_through_ball', 'kp_cut_back',
    ],
}


def feature_set(*tiers: str) -> list[str]:
    """Return the concatenated feature list for the requested tiers."""
    
    cols: list[str] = []
    for t in tiers:
        if t not in TIER_COLUMNS:
            raise KeyError(f'Unknown tier {t!r}; available: {list(TIER_COLUMNS)}')
        cols.extend(TIER_COLUMNS[t])
    return cols
