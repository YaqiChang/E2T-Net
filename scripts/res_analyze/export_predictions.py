"""Export sample-level prediction tables by calling eval.py once."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def run_eval_export(
    checkpoint: str,
    split: str,
    output_dir: str,
    output_name: str,
    artifact_dir: str = '',
    dataset: str = 'jaad',
    data_dir: str = '',
    out_dir: str = '',
    log_name: str = '',
) -> str:
    command = [
        sys.executable,
        os.path.join(REPO_ROOT, 'eval.py'),
        '--checkpoint', checkpoint,
        '--dtype', split,
        '--dataset', dataset,
        '--save_sample_results', 'True',
        '--sample_results_dir', output_dir,
        '--sample_results_name', output_name,
    ]
    if artifact_dir:
        command.extend(['--artifact_dir', artifact_dir])
    if data_dir:
        command.extend(['--data_dir', data_dir])
    if out_dir:
        command.extend(['--out_dir', out_dir])
    if log_name:
        command.extend(['--log_name', log_name])

    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return os.path.join(output_dir, output_name)
