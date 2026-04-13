import numpy as np

class PoseNPZAdapter:
    def __init__(self, pose_format="auto"):
        self.pose_format = pose_format

    def load(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)

        keys = list(data.keys())

        pose = None
        conf = None

        candidate_pose_keys = [
            "keypoints", "pose", "joints", "preds", "coords", "keypoints_2d"
        ]
        candidate_conf_keys = [
            "scores", "confidence", "conf", "keypoint_scores"
        ]

        for k in candidate_pose_keys:
            if k in keys:
                pose = data[k]
                break

        for k in candidate_conf_keys:
            if k in keys:
                conf = data[k]
                break

        if pose is None:
            raise KeyError(f"Cannot find pose array in {npz_path}. Keys: {keys}")

        pose = np.asarray(pose)

        if pose.ndim == 2:
            pose = pose[None]

        if pose.shape[-1] >= 3 and conf is None:
            conf = pose[..., 2]
            pose = pose[..., :2]

        if conf is None:
            conf = np.ones(pose.shape[:-1], dtype=np.float32)

        conf = np.asarray(conf).astype(np.float32)
        pose = np.asarray(pose).astype(np.float32)

        return pose, conf