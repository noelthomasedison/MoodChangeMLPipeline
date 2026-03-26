from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .config import ID_COL, MOOD_COLS, MOOD_GROUPS, PERSONAL_MODEL_MIN_POSITIVE_LABELS, PERSONAL_MODEL_MIN_ROWS, TRANSITION_PERSISTENCE


@dataclass
class ModelArtifact:
    scaler: Any
    model: dict[str, Any]
    labels: list[str]
    feature_cols: list[str]
    athlete_id: str | None = None


def _build_binary_model() -> LogisticRegression:
    return LogisticRegression(max_iter=800, class_weight="balanced", solver="liblinear")


def train_multilabel_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    labels: list[str],
    athlete_id: str | None = None,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> ModelArtifact:
    X = df[feature_cols].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    y = df[labels].astype(int)
    active_labels = [c for c in labels if y[c].sum() > 0]
    if not active_labels:
        raise ValueError("No positive mood labels available for training.")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    weights = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

    models: dict[str, LogisticRegression] = {}
    for lab in active_labels:
        clf = _build_binary_model()
        if weights is not None:
            clf.fit(Xs, y[lab], sample_weight=weights)
        else:
            clf.fit(Xs, y[lab])
        models[lab] = clf

    return ModelArtifact(scaler=scaler, model=models, labels=active_labels, feature_cols=feature_cols, athlete_id=athlete_id)


def predict_probabilities(artifact: ModelArtifact, df: pd.DataFrame, all_labels: list[str] | None = None) -> pd.DataFrame:
    labels = all_labels or artifact.labels
    X = df[artifact.feature_cols].astype(float).replace([np.inf, -np.inf], 0).fillna(0)
    Xs = artifact.scaler.transform(X)
    pred = pd.DataFrame(0.0, index=df.index, columns=labels)

    # Backward compatible: older artifacts stored a OneVsRestClassifier, newer ones store a dict of per-label models.
    if isinstance(artifact.model, dict):
        for lab in artifact.labels:
            clf = artifact.model[lab]
            pred[lab] = clf.predict_proba(Xs)[:, 1]
        return pred

    raw = artifact.model.predict_proba(Xs)
    if isinstance(raw, list):
        raw = np.column_stack(raw)
    for i, lab in enumerate(artifact.labels):
        pred[lab] = raw[:, i]
    return pred


def train_personal_models(df_train: pd.DataFrame, feature_cols: list[str]) -> dict[str, ModelArtifact]:
    models: dict[str, ModelArtifact] = {}
    for athlete_id, sub in df_train.groupby(ID_COL):
        if len(sub) < PERSONAL_MODEL_MIN_ROWS:
            continue
        positive_labels = int((sub[MOOD_COLS].sum(axis=0) > 0).sum())
        if positive_labels < PERSONAL_MODEL_MIN_POSITIVE_LABELS:
            continue
        try:
            weights = sub["row_reliability_weight"].values if "row_reliability_weight" in sub.columns else None
            models[str(athlete_id)] = train_multilabel_model(sub, feature_cols, MOOD_COLS, athlete_id=str(athlete_id), sample_weight=weights)
        except Exception:
            continue
    return models


def choose_artifact(athlete_id: str, personal_models: dict[str, ModelArtifact], global_model: ModelArtifact) -> ModelArtifact:
    return personal_models.get(str(athlete_id), global_model)


def mood_group_probabilities(prob_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=prob_df.index)
    for group, moods in MOOD_GROUPS.items():
        cols = [m for m in moods if m in prob_df.columns]
        out[group] = prob_df[cols].sum(axis=1) if cols else 0.0
    out["unknown"] = (out.sum(axis=1) == 0).astype(float)
    return out


def smooth_group_sequence(group_probs: pd.DataFrame, window: int = 3) -> pd.Series:
    smooth = group_probs.rolling(window=window, min_periods=1).mean()
    return smooth.idxmax(axis=1)


def detect_group_transitions(group_sequence: pd.Series, persistence: int = TRANSITION_PERSISTENCE) -> pd.Series:
    flags = pd.Series(0, index=group_sequence.index, dtype=int)
    values = group_sequence.tolist()
    idxs = list(group_sequence.index)
    for i in range(1, len(values)):
        if values[i] != values[i - 1]:
            future = values[i : min(len(values), i + persistence)]
            if len(future) >= persistence and all(v == values[i] for v in future):
                flags.loc[idxs[i]] = 1
    return flags


def top_k_hit_rate(y_true: pd.DataFrame, prob_df: pd.DataFrame, k: int = 3) -> float:
    hits = []
    for idx in y_true.index:
        true_labels = set(y_true.columns[(y_true.loc[idx] > 0).values])
        if not true_labels:
            continue
        top_labels = set(prob_df.loc[idx].sort_values(ascending=False).head(k).index.tolist())
        hits.append(int(bool(true_labels & top_labels)))
    return float(np.mean(hits)) if hits else 0.0


def evaluate_multilabel(y_true: pd.DataFrame, prob_df: pd.DataFrame, threshold: float = 0.35) -> dict[str, float]:
    y_pred = (prob_df >= threshold).astype(int)
    subset_acc = accuracy_score(y_true.values, y_pred.values)
    micro = f1_score(y_true.values, y_pred.values, average="micro", zero_division=0)
    macro = f1_score(y_true.values, y_pred.values, average="macro", zero_division=0)
    return {
        "subset_accuracy": round(float(subset_acc), 4),
        "micro_f1": round(float(micro), 4),
        "macro_f1": round(float(macro), 4),
        "top3_hit_rate": round(float(top_k_hit_rate(y_true, prob_df, 3)), 4),
    }


def transition_f1(true_group: pd.Series, pred_group: pd.Series) -> float:
    true_change = (true_group != true_group.shift(1)).astype(int).fillna(0)
    pred_change = (pred_group != pred_group.shift(1)).astype(int).fillna(0)
    return round(float(f1_score(true_change, pred_change, zero_division=0)), 4)
