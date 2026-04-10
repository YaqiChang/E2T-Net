"""Distribution comparison helpers for val/test result tables."""

from __future__ import annotations

import pandas as pd


def summarize_split_distribution(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return pd.DataFrame([{
        'split': split_name,
        'num_rows': int(len(df)),
        'num_sequences': int(df['sequence_key'].nunique()) if 'sequence_key' in df.columns else int(len(df)),
        'num_videos': int(df['video_id'].nunique()) if 'video_id' in df.columns else 0,
        'num_pedestrians': int(df['ped_id'].nunique()) if 'ped_id' in df.columns else 0,
        'state_positive_rate': float(df['state_gt'].mean()),
        'intent_positive_rate': float(df['intent_gt'].mean()),
        'state_score_mean': float(df['state_score'].mean()),
        'intent_score_mean': float(df['intent_score'].mean()),
    }])


def top_count_table(df: pd.DataFrame, column: str, split_name: str, top_k: int = 20) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=['split', column, 'count'])
    counts = df[column].value_counts().head(top_k).rename_axis(column).reset_index(name='count')
    counts.insert(0, 'split', split_name)
    return counts


def overlap_summary(val_df: pd.DataFrame, eval_df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in val_df.columns or column not in eval_df.columns:
        return pd.DataFrame([{'column': column, 'val_unique': 0, 'eval_unique': 0, 'overlap': 0}])

    val_set = set(val_df[column].dropna().tolist())
    eval_set = set(eval_df[column].dropna().tolist())
    return pd.DataFrame([{
        'column': column,
        'val_unique': len(val_set),
        'eval_unique': len(eval_set),
        'overlap': len(val_set.intersection(eval_set)),
    }])
