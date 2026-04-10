"""Plot helpers for threshold and bucket analysis."""

from __future__ import annotations

import os


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('matplotlib is required for plotting analysis results') from exc
    return plt


def plot_threshold_curves(df, output_path: str, title: str) -> str:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ['precision', 'recall', 'f1', 'balanced_accuracy']:
        if metric in df.columns:
            ax.plot(df['threshold'], df[metric], label=metric)
    ax.set_title(title)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_bucket_curves(df, output_path: str, title: str) -> str:
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    bucket_labels = df['time_bucket'].tolist()
    for metric in ['precision', 'recall', 'f1', 'balanced_accuracy']:
        if metric in df.columns:
            ax.plot(x, df[metric], marker='o', label=metric)
    ax.set_title(title)
    ax.set_xlabel('Time bucket')
    ax.set_ylabel('Metric')
    ax.set_xticks(list(x))
    ax.set_xticklabels(bucket_labels, rotation=30, ha='right')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
