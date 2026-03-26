from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import ARTIFACT_DIR, ID_COL, MOOD_COLS, RAW_DATA_PATH
from src.data import chronological_split, load_raw_data, prepare_features
from src.model import (
    choose_artifact,
    evaluate_multilabel,
    mood_group_probabilities,
    predict_probabilities,
    smooth_group_sequence,
    train_multilabel_model,
    train_personal_models,
    transition_f1,
)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw_data(str(RAW_DATA_PATH))
    prepared = prepare_features(raw)
    frame = prepared.frame.copy()
    frame["split"] = chronological_split(frame)

    train_df = frame[frame["split"] == "train"].copy()
    val_df = frame[frame["split"] == "val"].copy()
    test_df = frame[frame["split"] == "test"].copy()

    global_weights = train_df["row_reliability_weight"].values if "row_reliability_weight" in train_df.columns else None
    global_model = train_multilabel_model(train_df, prepared.feature_cols, MOOD_COLS, sample_weight=global_weights)
    personal_models = train_personal_models(train_df, prepared.feature_cols)

    def infer(df: pd.DataFrame) -> pd.DataFrame:
        chunks = []
        for athlete_id, sub in df.groupby(ID_COL):
            artifact = choose_artifact(str(athlete_id), personal_models, global_model)
            probs = predict_probabilities(artifact, sub, MOOD_COLS)
            probs.index = sub.index
            chunks.append(probs)
        return pd.concat(chunks).sort_index() if chunks else pd.DataFrame(columns=MOOD_COLS)

    val_probs = infer(val_df)
    test_probs = infer(test_df)

    val_metrics = evaluate_multilabel(val_df[MOOD_COLS].astype(int), val_probs) if not val_df.empty else {}
    test_metrics = evaluate_multilabel(test_df[MOOD_COLS].astype(int), test_probs) if not test_df.empty else {}

    def eval_transition(df: pd.DataFrame, probs: pd.DataFrame) -> float:
        scores = []
        for athlete_id, sub in df.groupby(ID_COL):
            p = probs.loc[sub.index]
            pred_group = smooth_group_sequence(mood_group_probabilities(p))
            true_group = sub["observed_mood_group"]
            scores.append(transition_f1(true_group, pred_group))
        return round(float(pd.Series(scores).mean()), 4) if scores else 0.0

    metrics = {
        "learning_type": "supervised multi-label classification with temporal persistence smoothing and reliability-aware weighting",
        "model_family": "One-vs-Rest Logistic Regression (weighted by row reliability)",
        "global_train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "personal_models_trained": int(len(personal_models)),
        "mean_train_row_weight": round(float(train_df["row_reliability_weight"].mean()), 4) if "row_reliability_weight" in train_df.columns else None,
        "validation": val_metrics,
        "test": test_metrics,
        "transition_f1_validation": eval_transition(val_df, val_probs) if not val_df.empty else 0.0,
        "transition_f1_test": eval_transition(test_df, test_probs) if not test_df.empty else 0.0,
    }

    joblib.dump(global_model, ARTIFACT_DIR / "global_model.joblib")
    joblib.dump(personal_models, ARTIFACT_DIR / "personal_models.joblib")
    (ARTIFACT_DIR / "feature_cols.json").write_text(json.dumps(prepared.feature_cols, indent=2))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
