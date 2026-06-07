from __future__ import annotations
import os
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from .features import build_features, feature_set, TIER_COLUMNS


DEFAULT_CSV = os.path.join('data', 'processed', 'all_shots.csv')


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------
def load_features(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    """Reads the processed CSV and returns the full engineered feature frame."""

    shots = pd.read_csv(csv_path)
    return build_features(shots)


def split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split on the goal label."""

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df['goal'],
        random_state=random_state,
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_proba) -> dict:
    """Creates all three calibration-aware metrics in one dict."""

    return {'auc_roc': float(roc_auc_score(y_true, y_proba)),
            'log_loss': float(log_loss(y_true, y_proba)),
            'brier': float(brier_score_loss(y_true, y_proba))}


def train_eval(model, train: pd.DataFrame, test: pd.DataFrame,
               feature_cols: Iterable[str], name: str = '',
               tier: str = '') -> dict:
    """Fit on train, score on test, returns a metrics row."""

    feature_cols = list(feature_cols)
    X_train, y_train = train[feature_cols], train['goal']
    X_test,  y_test  = test[feature_cols],  test['goal']

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    return {'model': name,
            'tier': tier,
            'n_features' : len(feature_cols),
            **evaluate(y_test, proba)}


# ---------------------------------------------------------------------------
# LR Non-Linear Features
# ---------------------------------------------------------------------------
LR_POLY_FEATURES = ['distance_sq', 'log_distance', 'dist_sin_angle', 'inv_distance']


def add_lr_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Classical hand-built xG transformations of distance + angle.

    Adds non-linear interactions to the linear logistic regression model.
    """

    df = df.copy()
    d = df['distance_to_goal']
    a = df['angle_to_goal']
    df['distance_sq'] = d ** 2
    df['log_distance'] = np.log1p(d)
    df['dist_sin_angle'] = d * np.sin(np.radians(a))
    df['inv_distance'] = 1.0 / (d + 1.0)
    return df
