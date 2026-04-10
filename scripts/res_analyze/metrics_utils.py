"""Metric and aggregation helpers for result analysis."""

from __future__ import annotations

from typing import Iterable, Sequence

import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import binary_classification_metrics


DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 101)
DEFAULT_BUCKET_EDGES = [-1e9, -30, -15, -5, 0, 5, 15, 30, 1e9]
DEFAULT_BUCKET_LABELS = [
    '<=-30',
    '(-30,-15]',
    '(-15,-5]',
    '(-5,0]',
    '(0,5]',
    '(5,15]',
    '(15,30]',
    '>30',
]


def compute_metrics_from_scores(
    scores: Iterable[float],
    targets: Iterable[int],
    threshold: float,
) -> dict:
    pred = (np.asarray(list(scores), dtype=np.float32) >= threshold).astype(np.int64)
    return binary_classification_metrics(pred, np.asarray(list(targets), dtype=np.int64))


def scan_thresholds(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    thresholds: Sequence[float] | None = None,
) -> pd.DataFrame:
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float32)
    rows = []
    scores = df[score_col].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.int64)
    for threshold in thresholds:
        metrics = compute_metrics_from_scores(scores, targets, float(threshold))
        rows.append({
            'threshold': float(threshold),
            **metrics,
        })
    return pd.DataFrame(rows)


def with_time_buckets(
    df: pd.DataFrame,
    bucket_edges: Sequence[float] | None = None,
    bucket_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    bucket_edges = DEFAULT_BUCKET_EDGES if bucket_edges is None else list(bucket_edges)
    bucket_labels = DEFAULT_BUCKET_LABELS if bucket_labels is None else list(bucket_labels)
    if len(bucket_edges) != len(bucket_labels) + 1:
        raise ValueError('bucket_edges length must equal len(bucket_labels) + 1')

    result = df.copy()
    result = result[result['time_to_trigger'].notna()].copy()
    result['time_to_trigger'] = result['time_to_trigger'].astype(float)
    result['time_bucket'] = pd.cut(
        result['time_to_trigger'],
        bins=bucket_edges,
        labels=bucket_labels,
        include_lowest=True,
        right=True,
    )
    return result


def summarize_by_bucket(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for bucket, bucket_df in df.groupby('time_bucket', observed=False):
        if bucket_df.empty:
            continue
        metrics = compute_metrics_from_scores(bucket_df[score_col], bucket_df[target_col], threshold)
        rows.append({
            'time_bucket': str(bucket),
            'num_rows': int(len(bucket_df)),
            'positive_rate': float(bucket_df[target_col].mean()),
            **metrics,
        })
    return pd.DataFrame(rows)


def per_sequence_intent_rows(df: pd.DataFrame) -> pd.DataFrame:
    if 'sequence_key' not in df.columns:
        raise ValueError('Intent analysis requires sequence_key column')
    return df.sort_values(['sequence_key', 'step_idx']).drop_duplicates('sequence_key', keep='first').copy()
