from __future__ import annotations

import math
from collections import defaultdict
import numpy as np
import pandas as pd

from .config import DRIVER_THEME_MAP, MAX_DRIVER_MISSING_RATE, MOOD_COLS, MOOD_ZONES, TYPE_COL
from .mappings import display_type, zone_for_mood

FEATURE_LABELS = {
    'Sleep': 'sleep',
    'sleep_hours': 'sleep',
    'Hydration': 'hydration',
    'Nutrition': 'nutrition',
    'HRV': 'HRV',
    'Resting Heart Rate': 'resting heart rate',
    'Sleep Heart Rate': 'sleep heart rate',
    'Tiredness': 'tiredness',
    'Performance': 'performance',
    'Concentrate': 'focus',
    'Steps': 'activity',
    'SorenessUp': 'upper body soreness',
    'SorenessLo': 'lower body soreness',
    'Social': 'social energy',
}

THEME_COPY = {
    'Recovery & Sleep': ('Recovery & Sleep', 'sleep and recovery are lower than usual'),
    'Body Signals': ('Body Signals', 'body-based signals suggest more strain than usual'),
    'Daily Load & Routine': ('Daily Load & Routine', 'daily load and routine look different from the athlete’s recent pattern'),
    'Emotional Carry-over': ('Emotional Carry-over', 'the previous check-in is still influencing how the athlete feels now'),
}


def summarize_moods(prob_row: pd.Series, observed_row: pd.Series | None = None, top_k: int = 3) -> list[str]:
    if observed_row is not None:
        chosen = [m for m in MOOD_COLS if int(observed_row.get(m, 0)) == 1]
        if chosen:
            return chosen[:top_k]
    return prob_row.sort_values(ascending=False).head(top_k).index.tolist()


def _base_feature_name(col: str) -> str:
    suffixes = ['_roll3', '_roll7', '_delta1', '_ema', '_z', '_missing_rate_5', '_imputed_rate_5', '_vs_type_baseline']
    for suf in suffixes:
        if col.endswith(suf):
            return col[:-len(suf)]
    return col


def _feature_theme(col: str) -> str:
    base = _base_feature_name(col)
    for theme, features in DRIVER_THEME_MAP.items():
        if col in features or base in features:
            return theme
    return 'Body Signals'


def _human_feature(col: str) -> str:
    return FEATURE_LABELS.get(_base_feature_name(col), _base_feature_name(col).replace('_', ' ').lower())


def compute_driver_scores(
    history: pd.DataFrame,
    latest_idx: int,
    feature_cols: list[str],
    artifact,
    predicted_mood: str,
) -> tuple[list[dict], list[dict]]:
    if latest_idx not in history.index:
        return [], []
    latest = history.loc[latest_idx]
    previous = history.loc[:latest_idx - 1].tail(5)
    if previous.empty:
        return [], []

    feature_to_coef = {}
    model = getattr(artifact, 'model', None)
    scaler = getattr(artifact, 'scaler', None)
    if isinstance(model, dict) and predicted_mood in model:
        clf = model[predicted_mood]
        coef = clf.coef_[0]
        for i, col in enumerate(artifact.feature_cols):
            feature_to_coef[col] = float(abs(coef[i]))

    feature_scores: list[dict] = []
    for col in feature_cols:
        if col not in history.columns:
            continue
        if col in MOOD_COLS:
            continue
        if pd.isna(latest[col]):
            continue
        missing_rate_col = f'{_base_feature_name(col)}_missing_rate_5'
        reliability = 1.0
        if missing_rate_col in history.columns:
            reliability = max(0.2, 1.0 - float(latest.get(missing_rate_col, 0.0)))
        if reliability < (1.0 - MAX_DRIVER_MISSING_RATE):
            continue
        prev_mean = previous[col].mean()
        prev_std = previous[col].std()
        prev_std = float(prev_std) if pd.notna(prev_std) and prev_std > 1e-6 else 1.0
        recent_change = float(latest[col] - prev_mean)
        recent_change_z = recent_change / prev_std
        model_influence = feature_to_coef.get(col, 0.5)
        score = abs(model_influence) * abs(recent_change_z) * reliability
        if score <= 0.02:
            continue
        feature_scores.append({
            'feature': col,
            'base_feature': _base_feature_name(col),
            'label': _human_feature(col),
            'theme': _feature_theme(col),
            'direction': 'higher' if recent_change > 0 else 'lower',
            'score': float(score),
            'recent_change_z': float(recent_change_z),
            'reliability': float(reliability),
        })

    feature_scores = sorted(feature_scores, key=lambda x: x['score'], reverse=True)

    theme_scores = defaultdict(lambda: {'score': 0.0, 'features': []})
    for item in feature_scores:
        theme_scores[item['theme']]['score'] += item['score']
        theme_scores[item['theme']]['features'].append(item)

    themed = []
    for theme, bucket in theme_scores.items():
        top_feature = sorted(bucket['features'], key=lambda x: x['score'], reverse=True)[0]
        title, default_copy = THEME_COPY.get(theme, (theme, 'this area looks different from the recent pattern'))
        direction_copy = 'lower than usual' if top_feature['direction'] == 'lower' else 'higher than usual'
        themed.append({
            'theme': title,
            'score': round(bucket['score'], 3),
            'headline': f"{top_feature['label'].capitalize()} is {direction_copy}",
            'detail': default_copy,
            'feature': top_feature['label'],
            'direction': top_feature['direction'],
        })
    themed = sorted(themed, key=lambda x: x['score'], reverse=True)[:3]
    return feature_scores[:10], themed


def build_summary(prev_mood: str | None, curr_mood: str | None, prev_zone: str, curr_zone: str, driver_themes: list[dict], context_type: str | None) -> str:
    context_text = display_type(context_type)
    if not curr_mood:
        return 'No clear mood signal is available yet.'

    if not prev_mood or prev_mood == 'No mood logged':
        if driver_themes:
            reasons = '; '.join(item['headline'] for item in driver_themes[:2])
            return f"The athlete is currently feeling {curr_mood} and is in a {curr_zone} state. This appears to be shaped by {reasons}. Context: {context_text}."
        return f"The athlete is currently feeling {curr_mood} and is in a {curr_zone} state. Context: {context_text}."

    if prev_mood == curr_mood:
        if driver_themes:
            support = '; '.join(item['headline'] for item in driver_themes[:2])
            return f"The athlete still feels {curr_mood} and remains in the {curr_zone} zone. This looks like emotional carry-over from the previous check-in, supported by {support}. Context: {context_text}."
        return f"The athlete still feels {curr_mood} and remains in the {curr_zone} zone. Context: {context_text}."

    if prev_zone == curr_zone:
        carry = f"The athlete has moved from {prev_mood} to {curr_mood}, but is still within the {curr_zone} zone."
    else:
        carry = f"The athlete has shifted from {prev_mood} ({prev_zone}) to {curr_mood} ({curr_zone}), which is a broader emotional change."

    if driver_themes:
        reasons = '; '.join(item['headline'] for item in driver_themes[:3])
        return f"{carry} The strongest influences right now appear to be {reasons}. Context: {context_text}."
    return f"{carry} Context: {context_text}."
