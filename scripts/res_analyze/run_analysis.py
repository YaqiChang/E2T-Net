"""Single entrypoint for result analysis experiments.

Modes:
- export: run eval.py once and save a sample-level prediction table
- threshold_scan: scan thresholds offline from an exported table
- time_bucket: aggregate metrics by time-to-trigger buckets
- distribution_compare: compare val and eval/test exported tables
- all: export then run all available analyses
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from distribution_analysis import overlap_summary, summarize_split_distribution, top_count_table
from export_predictions import run_eval_export
from io_utils import ensure_dir, join_output_path, read_prediction_table, write_table
from metrics_utils import (
    per_sequence_intent_rows,
    scan_thresholds,
    summarize_by_bucket,
    with_time_buckets,
)
from plot_utils import plot_bucket_curves, plot_threshold_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Run result analysis experiments from one entrypoint.')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['export', 'threshold_scan', 'time_bucket', 'distribution_compare', 'all'])
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--dataset', type=str, default='jaad')
    parser.add_argument('--artifact_dir', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--log_name', type=str, default='')
    parser.add_argument('--results_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--predictions_csv', type=str, default='')
    parser.add_argument('--val_predictions_csv', type=str, default='')
    parser.add_argument('--eval_predictions_csv', type=str, default='')
    parser.add_argument('--output_name', type=str, default='')
    parser.add_argument('--state_threshold', type=float, default=None)
    parser.add_argument('--intent_threshold', type=float, default=None)
    parser.add_argument('--skip_plots', action='store_true')
    return parser.parse_args()


def resolve_predictions_csv(args):
    filename = args.output_name or f'{args.dataset}_{args.split}_sample_predictions.csv'
    return args.predictions_csv or os.path.join(args.results_dir, filename)


def run_export(args):
    if not args.checkpoint:
        raise ValueError('--checkpoint is required for export mode')
    ensure_dir(args.results_dir)
    output_name = args.output_name or f'{args.dataset}_{args.split}_sample_predictions.csv'
    output_path = run_eval_export(
        checkpoint=args.checkpoint,
        split=args.split,
        output_dir=args.results_dir,
        output_name=output_name,
        artifact_dir=args.artifact_dir,
        dataset=args.dataset,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        log_name=args.log_name,
    )
    print(f'Exported sample results: {output_path}')
    return output_path


def run_threshold_scan(args, predictions_csv: str):
    df = read_prediction_table(predictions_csv)
    output_dir = ensure_dir(os.path.join(args.results_dir, 'threshold_scan'))

    state_table = scan_thresholds(df, 'state_score', 'state_gt')
    state_csv = write_table(state_table, join_output_path(output_dir, 'state_threshold_metrics.csv'))
    print(f'Saved state threshold scan: {state_csv}')

    intent_df = per_sequence_intent_rows(df)
    intent_table = scan_thresholds(intent_df, 'intent_score', 'intent_gt')
    intent_csv = write_table(intent_table, join_output_path(output_dir, 'intent_threshold_metrics.csv'))
    print(f'Saved intent threshold scan: {intent_csv}')

    if not args.skip_plots:
        plot_threshold_curves(state_table, join_output_path(output_dir, 'state_threshold_curves.png'), 'State Threshold Curves')
        plot_threshold_curves(intent_table, join_output_path(output_dir, 'intent_threshold_curves.png'), 'Intent Threshold Curves')


def run_time_bucket(args, predictions_csv: str):
    df = read_prediction_table(predictions_csv)
    bucketed_df = with_time_buckets(df)
    output_dir = ensure_dir(os.path.join(args.results_dir, 'time_bucket'))

    state_threshold = args.state_threshold if args.state_threshold is not None else float(df['state_threshold'].iloc[0])
    state_table = summarize_by_bucket(bucketed_df, 'state_score', 'state_gt', state_threshold)
    state_csv = write_table(state_table, join_output_path(output_dir, 'state_time_bucket_metrics.csv'))
    print(f'Saved state time-bucket summary: {state_csv}')

    intent_df = per_sequence_intent_rows(bucketed_df)
    intent_threshold = args.intent_threshold if args.intent_threshold is not None else float(intent_df['intent_threshold'].iloc[0])
    intent_table = summarize_by_bucket(intent_df, 'intent_score', 'intent_gt', intent_threshold)
    intent_csv = write_table(intent_table, join_output_path(output_dir, 'intent_time_bucket_metrics.csv'))
    print(f'Saved intent time-bucket summary: {intent_csv}')

    if not args.skip_plots:
        plot_bucket_curves(state_table, join_output_path(output_dir, 'state_time_bucket_curves.png'), 'State Time-Bucket Curves')
        plot_bucket_curves(intent_table, join_output_path(output_dir, 'intent_time_bucket_curves.png'), 'Intent Time-Bucket Curves')


def run_distribution_compare(args):
    if not args.val_predictions_csv or not args.eval_predictions_csv:
        raise ValueError('--val_predictions_csv and --eval_predictions_csv are required for distribution_compare mode')

    val_df = read_prediction_table(args.val_predictions_csv)
    eval_df = read_prediction_table(args.eval_predictions_csv)
    output_dir = ensure_dir(os.path.join(args.results_dir, 'distribution_compare'))

    summary = pd.concat(
        [
            summarize_split_distribution(val_df, 'val'),
            summarize_split_distribution(eval_df, 'eval'),
        ],
        ignore_index=True,
    )
    summary_csv = write_table(summary, join_output_path(output_dir, 'split_summary.csv'))
    print(f'Saved split summary: {summary_csv}')

    video_counts = pd.concat(
        [
            top_count_table(val_df, 'video_id', 'val'),
            top_count_table(eval_df, 'video_id', 'eval'),
        ],
        ignore_index=True,
    )
    write_table(video_counts, join_output_path(output_dir, 'top_video_counts.csv'))

    ped_counts = pd.concat(
        [
            top_count_table(val_df, 'ped_id', 'val'),
            top_count_table(eval_df, 'ped_id', 'eval'),
        ],
        ignore_index=True,
    )
    write_table(ped_counts, join_output_path(output_dir, 'top_ped_counts.csv'))

    overlap = pd.concat(
        [
            overlap_summary(val_df, eval_df, 'video_id'),
            overlap_summary(val_df, eval_df, 'ped_id'),
        ],
        ignore_index=True,
    )
    overlap_csv = write_table(overlap, join_output_path(output_dir, 'split_overlap_summary.csv'))
    print(f'Saved overlap summary: {overlap_csv}')


def main():
    args = parse_args()
    predictions_csv = resolve_predictions_csv(args)

    if args.mode in {'export', 'all'}:
        predictions_csv = run_export(args)

    if args.mode in {'threshold_scan', 'time_bucket', 'all'}:
        if not os.path.exists(predictions_csv):
            raise FileNotFoundError(f'Prediction csv not found: {predictions_csv}')

    if args.mode in {'threshold_scan', 'all'}:
        run_threshold_scan(args, predictions_csv)

    if args.mode in {'time_bucket', 'all'}:
        run_time_bucket(args, predictions_csv)

    if args.mode in {'distribution_compare', 'all'} and args.val_predictions_csv and args.eval_predictions_csv:
        run_distribution_compare(args)


if __name__ == '__main__':
    main()
