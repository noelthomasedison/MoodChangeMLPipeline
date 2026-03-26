from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .config import ID_COL, MOOD_ZONES, TYPE_COL

MALE_NAMES = ['Alex', 'Liam', 'Noah', 'Ethan', 'Mason', 'Leo', 'Ryan', 'Aiden', 'Lucas', 'Jordan', 'Sam', 'Owen', 'Caleb', 'Jay', 'Ben', 'Arjun', 'Aarav', 'Isaac', 'Milo']
FEMALE_NAMES = ['Sophia', 'Emma', 'Olivia', 'Ava', 'Mia', 'Ella', 'Zoe', 'Isla', 'Nina', 'Sara', 'Lily', 'Amelia', 'Layla', 'Freya', 'Ivy', 'Anya', 'Maya', 'Leah', 'Grace']
NEUTRAL_NAMES = ['Taylor', 'Riley', 'Casey', 'Jamie', 'Morgan', 'Avery', 'Quinn', 'Skyler', 'Reese', 'Parker']

TYPE_DISPLAY = {
    'pre_training': 'Before workout',
    'post_training': 'After workout',
    'end_of_day': 'End of day',
    'regular': 'Regular check-in',
    'unknown': 'General check-in',
}

@dataclass
class AthleteProfile:
    athlete_id: str
    display_name: str
    gender: str


def build_athlete_profiles(df: pd.DataFrame) -> dict[str, AthleteProfile]:
    profiles: dict[str, AthleteProfile] = {}
    male_i = female_i = neutral_i = 0
    for athlete_id, sub in df.groupby(ID_COL):
        gender_raw = str(sub['Gender'].dropna().iloc[0]).strip().lower() if 'Gender' in sub.columns and sub['Gender'].dropna().any() else 'unknown'
        if gender_raw.startswith('m'):
            name = MALE_NAMES[male_i % len(MALE_NAMES)]
            male_i += 1
            gender = 'Male'
        elif gender_raw.startswith('f'):
            name = FEMALE_NAMES[female_i % len(FEMALE_NAMES)]
            female_i += 1
            gender = 'Female'
        else:
            name = NEUTRAL_NAMES[neutral_i % len(NEUTRAL_NAMES)]
            neutral_i += 1
            gender = 'Athlete'
        profiles[str(athlete_id)] = AthleteProfile(str(athlete_id), name, gender)
    return profiles


def get_display_name(athlete_id: str, profiles: dict[str, AthleteProfile]) -> str:
    profile = profiles.get(str(athlete_id))
    return profile.display_name if profile else f'Athlete {str(athlete_id)[:4]}'


def zone_for_mood(mood: str | None) -> str:
    if not mood:
        return 'Unknown'
    return MOOD_ZONES.get(mood, 'Unknown')


def display_type(value: str | None) -> str:
    if not value:
        return TYPE_DISPLAY['unknown']
    return TYPE_DISPLAY.get(value, str(value).replace('_', ' ').title())
