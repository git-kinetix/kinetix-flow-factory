"""
RealisDance-Val Dataset Preparation for LTX Union NFT Training.

Usage (on RD-H200-2):
    python scripts/prepare_realisdance.py \
        --dataset-root /home/ubuntu/RealisDance-Val \
        --gvhmr-root /data/GVHMR \
        --output-jsonl /home/ubuntu/RealisDance-Val/train.jsonl \
        --output-joints /home/ubuntu/RealisDance-Val/gt_joints.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np


def scan_dataset(dataset_root: str) -> list[dict]:
    """Scan RealisDance-Val directory and build sample entries."""
    root = Path(dataset_root)
    samples = []

    video_dir = root / "videos"
    condition_dir = root / "condition" / "pose_depth"
    ref_image_dir = root / "ref_images"
    prompt_file = root / "prompts.json"

    prompts = {}
    if prompt_file.exists():
        with open(prompt_file) as f:
            prompts = json.load(f)

    if not video_dir.exists():
        print(f"Warning: {video_dir} not found, scanning root for videos")
        video_dir = root

    for video_path in sorted(video_dir.glob("*.mp4")):
        video_id = video_path.stem
        sample = {
            "video_id": video_id,
            "video": str(video_path.relative_to(root)),
            "prompt": prompts.get(video_id, f"A person performing an action"),
        }

        cond_path = condition_dir / f"{video_id}.mp4"
        if cond_path.exists():
            sample["reference_video"] = str(cond_path.relative_to(root))

        for ext in [".png", ".jpg", ".jpeg"]:
            ref_path = ref_image_dir / f"{video_id}{ext}"
            if ref_path.exists():
                sample["image"] = str(ref_path.relative_to(root))
                break

        samples.append(sample)

    return samples


def precompute_gt_joints(
    samples: list[dict],
    dataset_root: str,
    gvhmr_root: str,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Run GVHMR on all GT videos to extract COCO17 joints."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from DiffusionNFT.flow_grpo.gvhmr_pose_scorer import GVHMRPoseScorer

    scorer = GVHMRPoseScorer(device=device, gvhmr_root=gvhmr_root)
    gt_cache = {}

    for i, sample in enumerate(samples):
        video_id = sample["video_id"]
        video_path = os.path.join(dataset_root, sample["video"])
        print(f"[{i+1}/{len(samples)}] Processing {video_id}...")
        try:
            import imageio
            reader = imageio.get_reader(video_path)
            frames = []
            for frame in reader:
                t = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1.0
                frames.append(t)
            reader.close()
            video_tensor = torch.stack(frames, dim=1)
            joints = scorer._video_to_joints(video_tensor)
            gt_cache[video_id] = joints.cpu()
            print(f"  → {joints.shape[0]} frames, {joints.shape[1]} joints")
        except Exception as e:
            print(f"  → FAILED: {e}")
            continue

    return gt_cache


def main():
    parser = argparse.ArgumentParser(description="Prepare RealisDance-Val dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gvhmr-root", default="/data/GVHMR")
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-joints", default=None)
    parser.add_argument("--skip-joints", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.output_jsonl is None:
        args.output_jsonl = os.path.join(args.dataset_root, "train.jsonl")
    if args.output_joints is None:
        args.output_joints = os.path.join(args.dataset_root, "gt_joints.pt")

    print(f"Scanning {args.dataset_root}...")
    samples = scan_dataset(args.dataset_root)
    print(f"Found {len(samples)} samples")

    with open(args.output_jsonl, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"Written {args.output_jsonl}")

    if not args.skip_joints:
        print("Pre-computing GT SMPL joints...")
        gt_cache = precompute_gt_joints(samples, args.dataset_root, args.gvhmr_root, args.device)
        torch.save(gt_cache, args.output_joints)
        print(f"Saved {len(gt_cache)} joint caches to {args.output_joints}")
    else:
        print("Skipping joint pre-computation (--skip-joints)")


if __name__ == "__main__":
    main()
