# reward_server/mpjpe_server.py
"""
Remote MPJPE Reward Server.

Fallback for when GVHMR dependencies conflict with Flow-Factory's env.
Runs in the base conda env on RD-H200-2 where GVHMR is pre-installed.

Usage:
    conda activate base
    pip install fastapi uvicorn pillow
    python reward_server/mpjpe_server.py --port 8000
"""
import argparse
import io
import logging
import sys
import os

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MPJPE Reward Server")

scorer = None
gt_cache = None
SIGMA = 0.02


class RewardRequest(BaseModel):
    prompts: list[str]
    video_ids: list[str]


class RewardResponse(BaseModel):
    rewards: list[float]


@app.on_event("startup")
def load_models():
    global scorer, gt_cache
    from DiffusionNFT.flow_grpo.gvhmr_pose_scorer import GVHMRPoseScorer

    gvhmr_root = os.environ.get("GVHMR_ROOT", "/data/GVHMR")
    gt_joints_path = os.environ.get("GT_JOINTS_PATH", "/home/ubuntu/RealisDance-Val/gt_joints.pt")

    scorer = GVHMRPoseScorer(device="cuda", gvhmr_root=gvhmr_root)
    gt_cache = torch.load(gt_joints_path, weights_only=True)
    logger.info(f"Loaded GVHMR scorer and {len(gt_cache)} GT joint entries")


@app.post("/compute_reward", response_model=RewardResponse)
def compute_reward(request: RewardRequest):
    rewards = []
    for prompt, vid_id in zip(request.prompts, request.video_ids):
        try:
            if vid_id not in gt_cache:
                rewards.append(0.0)
                continue
            rewards.append(0.0)  # Placeholder — real impl receives video bytes
        except Exception as e:
            logger.warning(f"Failed for {vid_id}: {e}")
            rewards.append(0.0)
    return RewardResponse(rewards=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
