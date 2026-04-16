import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.jaad import JAAD
from model.pose_encoder import PoseSequenceEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for PoseSequenceEncoder")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="JAAD dataset root, e.g. /media/meta/File/datasets/Intention/JAAD_dataset/PN_ego",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/tmp/e2t_pose_stage2",
        help="Temporary output directory for cached JAAD sequence csv",
    )
    parser.add_argument("--dtype", type=str, default="train")
    parser.add_argument("--input", type=int, default=16)
    parser.add_argument("--output", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument(
        "--pose_file",
        type=str,
        default="",
        help="Optional explicit path to jaad_pose_annotations_fixed.npz",
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument(
        "--from_file",
        action="store_true",
        help="Use cached JAAD sequence csv if it already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Stage-2 scope:
    # 1. load real JAAD batch with pose and pose_conf
    # 2. run PoseSequenceEncoder only
    # 3. verify shape and numeric stability
    dataset = JAAD(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        dtype=args.dtype,
        input=args.input,
        output=args.output,
        stride=args.stride,
        skip=args.skip,
        from_file=args.from_file,
        save=True,
        use_images=False,
        use_attribute=False,
        use_opticalflow=False,
        use_pose=True,
        pose_file=args.pose_file,
        pose_format="jaad_hrnet_npz",
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    pose = batch["pose"].float()
    pose_conf = batch["pose_conf"].float()

    model = PoseSequenceEncoder(
        num_joints=pose.shape[2],
        frame_hidden_dim=args.hidden_dim,
        temporal_hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
    )
    model.eval()

    with torch.no_grad():
        pose_feat_seq = model(pose=pose, pose_conf=pose_conf)

    print("pose shape:", tuple(pose.shape))
    print("pose_conf shape:", tuple(pose_conf.shape))
    print("pose_feat_seq shape:", tuple(pose_feat_seq.shape))
    print("pose has NaN:", torch.isnan(pose).any().item())
    print("pose_conf has NaN:", torch.isnan(pose_conf).any().item())
    print("pose_feat_seq has NaN:", torch.isnan(pose_feat_seq).any().item())
    print("pose_feat_seq has Inf:", torch.isinf(pose_feat_seq).any().item())

    assert pose.ndim == 4
    assert pose_conf.ndim == 3
    assert pose_feat_seq.ndim == 3
    assert pose_feat_seq.shape[0] == pose.shape[0]
    assert pose_feat_seq.shape[1] == pose.shape[1]
    assert not torch.isnan(pose_feat_seq).any()
    assert not torch.isinf(pose_feat_seq).any()

    print("Stage 2 smoke test passed.")


if __name__ == "__main__":
    main()
