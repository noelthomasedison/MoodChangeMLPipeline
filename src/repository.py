from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH, ID_COL, MOOD_COLS, TIME_COL
from .data import load_raw_data

TABLE_NAME = "observations"


def _connect(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(raw_csv_path: str | Path, db_path: str | Path = DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLE_NAME}'")
    exists = cur.fetchone() is not None
    if not exists:
        df = load_raw_data(str(raw_csv_path))
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def read_all(db_path: str | Path = DB_PATH) -> pd.DataFrame:
    conn = _connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()
    if TIME_COL in df.columns:
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    return df


def get_athletes(db_path: str | Path = DB_PATH) -> list[str]:
    conn = _connect(db_path)
    rows = conn.execute(f"SELECT DISTINCT [{ID_COL}] AS athlete_id FROM {TABLE_NAME} ORDER BY athlete_id").fetchall()
    conn.close()
    return [str(r[0]) for r in rows]


def add_observation(observation: dict[str, Any], db_path: str | Path = DB_PATH) -> None:
    conn = _connect(db_path)
    df = pd.DataFrame([observation])
    for col in MOOD_COLS:
        if col not in df.columns:
            df[col] = False
    df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
