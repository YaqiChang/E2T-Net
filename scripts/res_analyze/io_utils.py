"""I/O helpers for result analysis.

This module keeps all result-table reading and writing logic in one place so
the main analysis flow stays small.
"""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def read_prediction_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f'Prediction table not found: {path}')

    df = pd.read_csv(path)
    required_columns = {
        'split',
        'sequence_key',
        'state_gt',
        'intent_gt',
        'state_score',
        'intent_score',
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f'Prediction table is missing required columns: {missing}')
    return df


def write_table(df: pd.DataFrame, output_path: str) -> str:
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    return output_path


def join_output_path(output_dir: str, name: str) -> str:
    ensure_dir(output_dir)
    return os.path.join(output_dir, name)


def load_multiple_tables(paths: Iterable[str]) -> list[pd.DataFrame]:
    return [read_prediction_table(path) for path in paths]
