from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.jaad import JAAD
from model.pose_encoder import PoseSequenceEncoder


def main():
    data_root = '/media/cyq/Data/dataset/Intention/JAAD_dataset/PN_ego'
    pose_file = str(Path(data_root) / 'jaad_pose_annotations_fixed.npz')
    cache_root = '/tmp/e2t_stage2_pose_encoder'
    Path(cache_root).mkdir(parents=True, exist_ok=True)

    dataset = JAAD(
        data_dir=data_root,
        out_dir=cache_root,
        dtype='train',
        input=16,
        output=32,
        stride=16,
        skip=1,
        from_file=False,
        save=True,
        use_images=False,
        use_attribute=False,
        use_opticalflow=False,
        use_pose=True,
        pose_file=pose_file,
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))

    pose = batch['pose']
    pose_conf = batch['pose_conf']

    encoder = PoseSequenceEncoder(num_joints=pose.shape[2], out_dim=128)
    pose_feat_seq = encoder(pose, pose_conf)

    print('pose shape:', tuple(pose.shape))
    print('pose_conf shape:', tuple(pose_conf.shape))
    print('pose_feat_seq shape:', tuple(pose_feat_seq.shape))

    assert pose_feat_seq.ndim == 3
    assert pose_feat_seq.shape[0] == pose.shape[0]
    assert pose_feat_seq.shape[1] == pose.shape[1]
    assert not torch.isnan(pose_feat_seq).any().item()
    assert not torch.isinf(pose_feat_seq).any().item()


if __name__ == '__main__':
    main()
