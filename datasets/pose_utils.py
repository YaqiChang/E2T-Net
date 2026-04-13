import os
from typing import Iterable, Tuple

import numpy as np


class JAADPoseNPZAdapter:
    """JAAD aggregated HRNet pose adapter.

    Standardized outputs:
    - pose: T x J x 2
    - pose_conf: T x J
    """

    REQUIRED_KEYS = ('ped_ids', 'ped_ptr', 'video_id', 'frame', 'keypoints')

    def __init__(self, pose_file: str, pose_format: str = 'jaad_hrnet_npz') -> None:
        if pose_format != 'jaad_hrnet_npz':
            raise ValueError(f'Unsupported JAAD pose format: {pose_format}')
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f'JAAD pose file not found: {pose_file}')

        self.pose_file = pose_file
        self.pose_format = pose_format
        self.index = {}
        self.missing_frames = 0
        self.total_frames = 0
        self.missing_sequences = 0

        payload = np.load(pose_file, allow_pickle=True)
        missing_keys = [key for key in self.REQUIRED_KEYS if key not in payload.files]
        if missing_keys:
            raise KeyError(
                f'JAAD pose file is missing keys {missing_keys}: {pose_file}'
            )

        keypoints = np.asarray(payload['keypoints'], dtype=np.float32)
        if keypoints.ndim != 3 or keypoints.shape[-1] < 3:
            raise ValueError(
                'Expected JAAD keypoints with shape N x J x 3, '
                f'got {tuple(keypoints.shape)} from {pose_file}'
            )

        self.pose = keypoints[..., :2]
        self.pose_conf = keypoints[..., 2]
        self.num_joints = int(self.pose.shape[1])

        video_ids = payload['video_id'].astype(str)
        frames = payload['frame'].astype(np.int64)
        ped_ids = self._expand_ped_ids(payload, len(video_ids))

        for row_index, (video_id, ped_id, frame_id) in enumerate(zip(video_ids, ped_ids, frames)):
            self.index[(str(video_id), str(ped_id), int(frame_id))] = row_index

    def _expand_ped_ids(self, payload: np.lib.npyio.NpzFile, total_records: int) -> np.ndarray:
        ped_ids = payload['ped_ids'].astype(str)
        ped_ptr = np.asarray(payload['ped_ptr'], dtype=np.int64)
        if ped_ptr.shape[0] != ped_ids.shape[0] + 1:
            raise ValueError(
                'ped_ptr length must equal ped_ids length + 1, '
                f'got ped_ptr={ped_ptr.shape[0]} ped_ids={ped_ids.shape[0]}'
            )

        expanded = np.empty((total_records,), dtype=object)
        for ped_index, ped_id in enumerate(ped_ids):
            start = int(ped_ptr[ped_index])
            end = int(ped_ptr[ped_index + 1])
            expanded[start:end] = ped_id
        return expanded.astype(str)

    def get_sequence(self, video_id: str, ped_id: str, frame_ids: Iterable[int]) -> Tuple[np.ndarray, np.ndarray, int]:
        frame_ids = [int(frame_id) for frame_id in frame_ids]
        pose = np.zeros((len(frame_ids), self.num_joints, 2), dtype=np.float32)
        pose_conf = np.zeros((len(frame_ids), self.num_joints), dtype=np.float32)

        missing_count = 0
        for time_index, frame_id in enumerate(frame_ids):
            row_index = self.index.get((str(video_id), str(ped_id), int(frame_id)))
            if row_index is None:
                missing_count += 1
                continue
            pose[time_index] = self.pose[row_index]
            pose_conf[time_index] = self.pose_conf[row_index]

        self.total_frames += len(frame_ids)
        self.missing_frames += missing_count
        if missing_count > 0:
            self.missing_sequences += 1

        return pose, pose_conf, missing_count

    def missing_summary(self) -> str:
        if self.total_frames == 0:
            return '0/0'
        return f'{self.missing_frames}/{self.total_frames}'
