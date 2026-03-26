from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import (
    ID_COL, INPUT_COLS, MAX_DRIVER_MISSING_RATE, MOOD_COLS, MOOD_GROUPS,
    SHORT_BFILL_LIMIT, SHORT_FFILL_LIMIT, TIME_COL, TYPE_COL, TYPE_VALUES,
)


@dataclass
class PreparedData:
    frame: pd.DataFrame
    feature_cols: list[str]
    target_cols: list[str]


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce', utc=True)
    df = df.dropna(subset=[ID_COL, TIME_COL]).sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.lower().replace({'man': 'men', 'woman': 'women'})
    if TYPE_COL in df.columns:
        df[TYPE_COL] = df[TYPE_COL].astype(str).replace({'nan': np.nan})
    for col in MOOD_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def dominant_moods(row: pd.Series) -> list[str]:
    return [m for m in MOOD_COLS if bool(row.get(m, False))]


def assign_group_from_row(row: pd.Series) -> str:
    scores = {group: int(row[moods].sum()) for group, moods in MOOD_GROUPS.items()}
    best_group = max(scores, key=scores.get)
    return best_group if scores[best_group] > 0 else 'unknown'


def _bucket_hour(ts: pd.Timestamp) -> int:
    if pd.isna(ts):
        return -1
    return int(pd.Timestamp(ts).tz_convert(None).hour)


def impute_type_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work['type_imputed'] = 0
    work['type_source'] = 'observed'
    work[TYPE_COL] = work[TYPE_COL].replace({'pre-workout': 'pre_training', 'post-workout': 'post_training'})
    work['hour_bucket'] = work[TIME_COL].apply(_bucket_hour)

    global_mode = (
        work.dropna(subset=[TYPE_COL])
            .groupby('hour_bucket')[TYPE_COL]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
            .to_dict()
    )

    pieces = []
    for athlete_id, sub in work.groupby(ID_COL):
        sub = sub.sort_values(TIME_COL).copy()
        athlete_mode = (
            sub.dropna(subset=[TYPE_COL])
               .groupby('hour_bucket')[TYPE_COL]
               .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
               .to_dict()
        )
        known = sub.dropna(subset=[TYPE_COL])
        for idx, row in sub[sub[TYPE_COL].isna()].iterrows():
            hb = row['hour_bucket']
            chosen = None
            source = None
            if hb in athlete_mode and pd.notna(athlete_mode[hb]):
                chosen = athlete_mode[hb]
                source = 'athlete_time_pattern'
            elif not known.empty:
                nearest = (known['hour_bucket'] - hb).abs().sort_values().index[0]
                chosen = known.loc[nearest, TYPE_COL]
                source = 'athlete_nearest_time'
            elif hb in global_mode and pd.notna(global_mode[hb]):
                chosen = global_mode[hb]
                source = 'global_time_pattern'
            else:
                chosen = 'unknown'
                source = 'unknown'
            sub.at[idx, TYPE_COL] = chosen
            sub.at[idx, 'type_imputed'] = 1
            sub.at[idx, 'type_source'] = source
        pieces.append(sub)
    out = pd.concat(pieces).sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    out[TYPE_COL] = out[TYPE_COL].fillna('unknown')
    out['type_missing'] = (out['type_source'] != 'observed').astype(int)
    return out


def _athlete_impute_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    original_missing = series.isna()
    filled = series.ffill(limit=SHORT_FFILL_LIMIT).bfill(limit=SHORT_BFILL_LIMIT)
    rolling_est = filled.rolling(3, min_periods=1).mean()
    filled = filled.fillna(rolling_est)
    athlete_median = filled.median(skipna=True)
    if pd.isna(athlete_median):
        athlete_median = 0.0
    filled = filled.fillna(athlete_median).fillna(0.0)
    imputed = (original_missing & filled.notna()).astype(int)
    return filled.astype(float), imputed


def prepare_features(df: pd.DataFrame) -> PreparedData:
    work = impute_type_hierarchical(df.copy())
    work = work.sort_values([ID_COL, TIME_COL]).reset_index(drop=True)
    work['date'] = work[TIME_COL].dt.tz_convert(None).dt.floor('D')

    numeric_cols = [c for c in INPUT_COLS if c in work.columns]
    raw_missing_cols: list[str] = []
    imputed_flag_cols: list[str] = []
    for col in numeric_cols:
        work[f'{col}_missing'] = work[col].isna().astype(int)
        work[f'{col}_imputed'] = 0
        raw_missing_cols.append(f'{col}_missing')
        imputed_flag_cols.append(f'{col}_imputed')

    def _per_user(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(TIME_COL).copy()
        for col in numeric_cols:
            g[col], g[f'{col}_imputed'] = _athlete_impute_series(g[col])
            g[f'{col}_roll3'] = g[col].rolling(3, min_periods=1).mean()
            g[f'{col}_roll7'] = g[col].rolling(7, min_periods=1).mean()
            g[f'{col}_delta1'] = g[col].diff().fillna(0)
            g[f'{col}_ema'] = g[col].ewm(span=3, adjust=False).mean()
            mean = g[col].expanding().mean()
            std = g[col].expanding().std().replace(0, np.nan)
            g[f'{col}_z'] = ((g[col] - mean) / std).replace([np.inf, -np.inf], 0).fillna(0)
            g[f'{col}_missing_rate_5'] = g[f'{col}_missing'].rolling(5, min_periods=1).mean()
            g[f'{col}_imputed_rate_5'] = g[f'{col}_imputed'].rolling(5, min_periods=1).mean()
            type_baseline = g.groupby(TYPE_COL)[col].transform('mean')
            g[f'{col}_vs_type_baseline'] = (g[col] - type_baseline).fillna(0)
        g['days_since_start'] = (g[TIME_COL] - g[TIME_COL].min()).dt.total_seconds() / 86400.0
        return g

    work = work.groupby(ID_COL, group_keys=False).apply(_per_user).reset_index(drop=True)

    work['sleep_hours'] = work['Sleep'] / 3600.0
    work['sleep_hours_roll3'] = work.groupby(ID_COL)['sleep_hours'].transform(lambda s: s.rolling(3, min_periods=1).mean())
    work['hrv_hr_ratio'] = (work['HRV'] / work['Resting Heart Rate'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)
    work['readiness_proxy'] = (
        0.22 * work['Performance'].fillna(0)
        + 0.18 * work['Concentrate'].fillna(0)
        + 0.15 * work['Hydration'].fillna(0)
        + 0.15 * work['Nutrition'].fillna(0)
        + 0.18 * work['sleep_hours_roll3'].fillna(0) * 10
        + 0.12 * work['HRV'].fillna(0)
        - 0.18 * work['Tiredness'].fillna(0)
        - 0.10 * work['SorenessUp'].fillna(0)
        - 0.10 * work['SorenessLo'].fillna(0)
    )
    work['dominant_observed_mood'] = work.apply(lambda r: dominant_moods(r)[0] if dominant_moods(r) else 'No mood logged', axis=1)
    work['observed_mood_group'] = work.apply(assign_group_from_row, axis=1)
    work['prev_primary_mood'] = work.groupby(ID_COL)['dominant_observed_mood'].shift(1).fillna('No mood logged')

    work['num_key_features_missing'] = work[raw_missing_cols].sum(axis=1)
    work['num_key_features_imputed'] = work[imputed_flag_cols].sum(axis=1)
    work['missing_ratio'] = work['num_key_features_missing'] / max(len(numeric_cols), 1)
    work['imputed_ratio'] = work['num_key_features_imputed'] / max(len(numeric_cols), 1)
    work['row_reliability_weight'] = (1.0 - 0.7 * work['imputed_ratio']).clip(lower=0.3, upper=1.0)

    if 'Gender' in work.columns:
        gender_dummies = pd.get_dummies(work['Gender'].fillna('unknown'), prefix='gender', dtype=int)
    else:
        gender_dummies = pd.DataFrame(index=work.index)
    type_dummies = pd.get_dummies(work[TYPE_COL].fillna('unknown'), prefix='type', dtype=int)
    prev_mood_dummies = pd.get_dummies(work['prev_primary_mood'].fillna('No mood logged'), prefix='prev_mood', dtype=int)
    work = pd.concat([work, gender_dummies, type_dummies, prev_mood_dummies], axis=1)

    feature_cols = []
    for col in numeric_cols:
        feature_cols.extend([
            col, f'{col}_missing', f'{col}_imputed', f'{col}_roll3', f'{col}_roll7', f'{col}_delta1', f'{col}_ema',
            f'{col}_z', f'{col}_missing_rate_5', f'{col}_imputed_rate_5', f'{col}_vs_type_baseline'
        ])
    feature_cols.extend([
        'sleep_hours', 'sleep_hours_roll3', 'hrv_hr_ratio', 'readiness_proxy', 'days_since_start', 'type_imputed', 'type_missing',
        'num_key_features_missing', 'num_key_features_imputed', 'missing_ratio', 'imputed_ratio'
    ])
    feature_cols.extend(gender_dummies.columns.tolist())
    feature_cols.extend(type_dummies.columns.tolist())
    feature_cols.extend(prev_mood_dummies.columns.tolist())
    feature_cols = [c for c in feature_cols if c in work.columns]

    work[feature_cols] = work[feature_cols].replace([np.inf, -np.inf], 0).fillna(0)
    work[MOOD_COLS] = work[MOOD_COLS].astype(int)
    return PreparedData(frame=work, feature_cols=feature_cols, target_cols=MOOD_COLS)


def chronological_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> pd.Series:
    split = pd.Series(index=df.index, dtype='object')
    for _, sub in df.groupby(ID_COL):
        idx = sub.sort_values(TIME_COL).index.tolist()
        n = len(idx)
        if n < 5:
            for i in idx:
                split.loc[i] = 'train'
            continue
        train_end = max(1, int(n * train_ratio))
        val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
        for i, row_idx in enumerate(idx):
            split.loc[row_idx] = 'train' if i < train_end else 'val' if i < val_end else 'test'
    return split.fillna('train')
