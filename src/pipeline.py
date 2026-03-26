from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd

from .config import ARTIFACT_DIR, ID_COL, MIN_TREND_POINTS, MAX_TREND_IMPUTED_RATIO, MOOD_COLS, RAW_DATA_PATH, TYPE_COL
from .data import prepare_features
from .explain import build_summary, compute_driver_scores, summarize_moods
from .mappings import build_athlete_profiles, get_display_name, zone_for_mood, display_type
from .model import choose_artifact, detect_group_transitions, mood_group_probabilities, predict_probabilities, smooth_group_sequence
from .repository import initialize_database, read_all


def load_artifacts(artifact_dir: str | Path = ARTIFACT_DIR) -> dict:
    artifact_dir = Path(artifact_dir)
    return {
        'global_model': joblib.load(artifact_dir / 'global_model.joblib'),
        'personal_models': joblib.load(artifact_dir / 'personal_models.joblib'),
        'metrics': json.loads((artifact_dir / 'metrics.json').read_text()),
        'feature_cols': json.loads((artifact_dir / 'feature_cols.json').read_text()),
    }


def assess_trend_quality(history: pd.DataFrame, value_col: str, imputed_col: str | None = None) -> dict:
    if value_col not in history.columns:
        return {'show': False, 'reason': f'{value_col} is unavailable.'}
    series = history[value_col]
    valid_points = int(series.notna().sum())
    unique_points = int(series.dropna().nunique())
    imputed_ratio = float(history[imputed_col].mean()) if imputed_col and imputed_col in history.columns else 0.0
    if valid_points < MIN_TREND_POINTS:
        return {'show': False, 'reason': f'Not enough reliable {value_col.lower()} data to display a trend.'}
    if unique_points <= 1:
        return {'show': False, 'reason': f'{value_col} has too little variation to show a meaningful trend.'}
    if imputed_ratio > MAX_TREND_IMPUTED_RATIO:
        return {'show': False, 'reason': f'Too much of the recent {value_col.lower()} data is imputed to display a trustworthy trend.'}
    return {'show': True, 'reason': ''}


def _build_history(frame: pd.DataFrame, probs: pd.DataFrame, group_sequence: pd.Series, transition_flags: pd.Series) -> pd.DataFrame:
    history_cols = [ID_COL, 'date', TYPE_COL, 'dominant_observed_mood', 'prev_primary_mood', 'Sleep', 'Hydration', 'Nutrition', 'HRV', 'Resting Heart Rate', 'Sleep Heart Rate', 'Tiredness', 'Performance', 'Concentrate', 'Steps', 'sleep_hours', 'Sleep_imputed', 'HRV_imputed', 'row_reliability_weight', 'imputed_ratio']
    history_cols = [c for c in history_cols if c in frame.columns]
    history = frame[history_cols].copy()
    history['predicted_primary_mood'] = probs.idxmax(axis=1)
    history['predicted_zone'] = history['predicted_primary_mood'].map(zone_for_mood)
    history['predicted_group'] = group_sequence
    history['transition_flag'] = transition_flags
    history['transition_score'] = mood_group_probabilities(probs).max(axis=1)
    return history


def refresh_payload(selected_athlete: str | None = None) -> dict:
    initialize_database(RAW_DATA_PATH)
    artifacts = load_artifacts()
    raw = read_all()
    profiles = build_athlete_profiles(raw)
    prepared = prepare_features(raw)
    all_frame = prepared.frame.copy()

    athlete_ids = sorted(all_frame[ID_COL].astype(str).unique().tolist())
    if not athlete_ids:
        return {'athletes': [], 'empty': True}

    athlete_id = selected_athlete if selected_athlete in athlete_ids else athlete_ids[0]
    frame = all_frame[all_frame[ID_COL].astype(str) == str(athlete_id)].copy()
    artifact = choose_artifact(str(athlete_id), artifacts['personal_models'], artifacts['global_model'])
    probs = predict_probabilities(artifact, frame, MOOD_COLS)
    probs.index = frame.index
    group_probs = mood_group_probabilities(probs)
    group_sequence = smooth_group_sequence(group_probs)
    transition_flags = detect_group_transitions(group_sequence)
    history = _build_history(frame, probs, group_sequence, transition_flags)

    latest_idx = history.index[-1]
    prev_idx = history.index[-2] if len(history) >= 2 else latest_idx
    curr_moods = summarize_moods(probs.loc[latest_idx], observed_row=frame.loc[latest_idx])
    prev_moods = summarize_moods(probs.loc[prev_idx], observed_row=frame.loc[prev_idx]) if len(history) >= 2 else []
    curr_mood = curr_moods[0] if curr_moods else None
    prev_mood = prev_moods[0] if prev_moods else None
    curr_zone = zone_for_mood(curr_mood)
    prev_zone = zone_for_mood(prev_mood)

    feature_scores, driver_themes = compute_driver_scores(frame, latest_idx, artifact.feature_cols, artifact, curr_mood or '')
    summary = build_summary(prev_mood, curr_mood, prev_zone, curr_zone, driver_themes, frame.loc[latest_idx].get(TYPE_COL))

    coach_rows = []
    for aid, sub in all_frame.groupby(ID_COL):
        art = choose_artifact(str(aid), artifacts['personal_models'], artifacts['global_model'])
        p = predict_probabilities(art, sub.tail(8), MOOD_COLS)
        pred = p.idxmax(axis=1)
        current = pred.iloc[-1]
        previous = pred.iloc[-2] if len(pred) > 1 else current
        current_zone = zone_for_mood(current)
        change = 'Mood shift' if current != previous else 'Stable'
        priority = 2 if current_zone == 'Stressed & Overloaded' else 1 if current_zone == 'Drained & Low Energy' else 0
        coach_rows.append({
            'athlete_id': str(aid),
            'athlete_name': get_display_name(str(aid), profiles),
            'current_mood': current,
            'current_zone': current_zone,
            'previous_mood': previous,
            'change': change,
            'priority': priority,
            'updated': str(sub['date'].iloc[-1].date()),
        })
    coach_df = pd.DataFrame(coach_rows).sort_values(['priority', 'athlete_name'], ascending=[False, True]).reset_index(drop=True)

    zone_snapshot = coach_df['current_zone'].value_counts().to_dict() if not coach_df.empty else {}
    alerts = coach_df[coach_df['priority'] > 0].head(6).to_dict('records') if not coach_df.empty else []

    return {
        'athlete_id': athlete_id,
        'athlete_name': get_display_name(str(athlete_id), profiles),
        'athletes': athlete_ids,
        'athlete_options': {aid: get_display_name(str(aid), profiles) for aid in athlete_ids},
        'history': history.reset_index(drop=True),
        'summary': summary,
        'current_mood': curr_mood,
        'previous_mood': prev_mood,
        'current_zone': curr_zone,
        'previous_zone': prev_zone,
        'current_context': display_type(frame.loc[latest_idx].get(TYPE_COL)),
        'driver_themes': driver_themes,
        'feature_scores': feature_scores,
        'metrics': artifacts['metrics'],
        'trend_quality': {
            'sleep': assess_trend_quality(history, 'sleep_hours', 'Sleep_imputed' if 'Sleep_imputed' in history.columns else None),
            'hrv': assess_trend_quality(history, 'HRV', 'HRV_imputed' if 'HRV_imputed' in history.columns else None),
        },
        'coach_df': coach_df,
        'alerts': alerts,
        'zone_snapshot': zone_snapshot,
    }
