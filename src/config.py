from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
RAW_DATA_PATH = DATA_DIR / 'raw' / 'Prorizon_user_data_2026-03-13.csv'
ARTIFACT_DIR = DATA_DIR / 'artifacts'
DB_PATH = DATA_DIR / 'app.db'

ID_COL = 'User ID'
TIME_COL = 'Date/Time'
TYPE_COL = 'Type'

MOOD_COLS = [
    'Overwhelmed', 'Angry', 'Fearful', 'Frustrated', 'Stressed', 'Restless', 'Worried', 'Irritable',
    'Determined', 'Excited', 'Empowered', 'Joyful', 'Motivated', 'Proud', 'Hopeful', 'Happy',
    'Shame', 'Disappointed', 'Sad', 'Bored', 'Lonely', 'Fatigued', 'Depressed', 'Drained',
    'Secure', 'Content', 'Calm', 'Grateful', 'Relaxed', 'Serene', 'Comfortable', 'Peaceful',
]

INPUT_COLS = [
    'Performance', 'Tiredness', 'Concentrate', 'Social', 'SorenessUp', 'SorenessLo', 'Hydration', 'Nutrition',
    'Steps', 'Sleep', 'HRV', 'Sleep Heart Rate', 'Resting Heart Rate', 'Weight', 'Height'
]

TYPE_VALUES = ['pre_training', 'post_training', 'end_of_day', 'regular', 'unknown']
SHORT_FFILL_LIMIT = 2
SHORT_BFILL_LIMIT = 1
PERSONAL_MODEL_MIN_ROWS = 45
PERSONAL_MODEL_MIN_POSITIVE_LABELS = 2
TRANSITION_PERSISTENCE = 2
MAX_DRIVER_MISSING_RATE = 0.5
MIN_TREND_POINTS = 4
MAX_TREND_IMPUTED_RATIO = 0.5

MOOD_ZONES = {
    'Overwhelmed': 'Stressed & Overloaded',
    'Angry': 'Stressed & Overloaded',
    'Fearful': 'Stressed & Overloaded',
    'Frustrated': 'Stressed & Overloaded',
    'Stressed': 'Stressed & Overloaded',
    'Restless': 'Stressed & Overloaded',
    'Worried': 'Stressed & Overloaded',
    'Irritable': 'Stressed & Overloaded',
    'Determined': 'Energised & Driven',
    'Excited': 'Energised & Driven',
    'Empowered': 'Energised & Driven',
    'Joyful': 'Energised & Driven',
    'Motivated': 'Energised & Driven',
    'Proud': 'Energised & Driven',
    'Hopeful': 'Energised & Driven',
    'Happy': 'Energised & Driven',
    'Shame': 'Drained & Low Energy',
    'Disappointed': 'Drained & Low Energy',
    'Sad': 'Drained & Low Energy',
    'Bored': 'Drained & Low Energy',
    'Lonely': 'Drained & Low Energy',
    'Fatigued': 'Drained & Low Energy',
    'Depressed': 'Drained & Low Energy',
    'Drained': 'Drained & Low Energy',
    'Secure': 'Calm & Balanced',
    'Content': 'Calm & Balanced',
    'Calm': 'Calm & Balanced',
    'Grateful': 'Calm & Balanced',
    'Relaxed': 'Calm & Balanced',
    'Serene': 'Calm & Balanced',
    'Comfortable': 'Calm & Balanced',
    'Peaceful': 'Calm & Balanced',
}

ZONE_ORDER = ['Energised & Driven', 'Calm & Balanced', 'Stressed & Overloaded', 'Drained & Low Energy']

MOOD_GROUPS = {
    'stressed_negative': ['Overwhelmed', 'Angry', 'Fearful', 'Frustrated', 'Stressed', 'Restless', 'Worried', 'Irritable', 'Shame', 'Disappointed', 'Sad', 'Depressed'],
    'positive_engaged': ['Determined', 'Excited', 'Empowered', 'Joyful', 'Motivated', 'Proud', 'Hopeful', 'Happy'],
    'low_energy': ['Bored', 'Lonely', 'Fatigued', 'Drained'],
    'calm_recovered': ['Secure', 'Content', 'Calm', 'Grateful', 'Relaxed', 'Serene', 'Comfortable', 'Peaceful'],
}

DRIVER_THEME_MAP = {
    'Recovery & Sleep': ['Sleep', 'sleep_hours', 'sleep_hours_roll3', 'Tiredness', 'SorenessUp', 'SorenessLo'],
    'Body Signals': ['HRV', 'Sleep Heart Rate', 'Resting Heart Rate', 'Hydration', 'Nutrition', 'hrv_hr_ratio'],
    'Daily Load & Routine': ['Steps', 'Performance', 'Concentrate', 'Social', 'days_since_start', TYPE_COL],
    'Emotional Carry-over': ['prev_primary_mood', 'prev_zone'],
}
