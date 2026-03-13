"""
MPJPE Reward Model for pose-conditioned video generation.

Computes pelvis-aligned MPJPE between SMPL poses predicted by GVHMR
on ground-truth videos (pre-computed, cached) vs. generated videos.
Reward = exp(-mpjpe / sigma) where sigma=0.02m.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class MPJPERewardModel(PointwiseRewardModel):
    """Pointwise reward based on pelvis-aligned MPJPE."""

    required_fields = ("prompt", "video")
    use_tensor_inputs = True

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)
        self.sigma = 0.02

        self._scorer = None
        self._gvhmr_root = getattr(config, "gvhmr_root", "/data/GVHMR")

        gt_path = getattr(config, "gt_joints_path", None)
        if gt_path:
            self.gt_cache = torch.load(gt_path, weights_only=True)
            logger.info(f"Loaded GT joint cache: {len(self.gt_cache)} entries from {gt_path}")
        else:
            self.gt_cache = {}
            logger.warning("No gt_joints_path provided — reward will be 0.0 for all samples")

    @property
    def scorer(self):
        if self._scorer is None:
            from DiffusionNFT.flow_grpo.gvhmr_pose_scorer import GVHMRPoseScorer
            self._scorer = GVHMRPoseScorer(
                device=self.device, gvhmr_root=self._gvhmr_root
            )
        return self._scorer

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[torch.Tensor]] = None,
        condition_images: Optional[List] = None,
        condition_videos: Optional[List] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if video is None:
            return RewardModelOutput(
                rewards=torch.zeros(len(prompt), device=self.device)
            )

        video_ids = kwargs.get("video_id", [None] * len(prompt))

        rewards = []
        for vid_tensor, vid_id in zip(video, video_ids):
            try:
                if vid_id is None or vid_id not in self.gt_cache:
                    logger.warning(f"No GT joints for video_id={vid_id}, reward=0.0")
                    rewards.append(0.0)
                    continue

                gt_joints = self.gt_cache[vid_id]

                vid_input = vid_tensor * 2.0 - 1.0
                if vid_input.dim() == 4 and vid_input.shape[1] == 3:
                    vid_input = vid_input.permute(1, 0, 2, 3)

                gen_joints = self.scorer._video_to_joints(vid_input)
                T = min(gen_joints.shape[0], gt_joints.shape[0])

                from DiffusionNFT.flow_grpo.gvhmr_pose_scorer import GVHMRPoseScorer
                mpjpe = GVHMRPoseScorer._pelvis_aligned_mpjpe(
                    gen_joints[:T], gt_joints[:T].to(gen_joints.device)
                )
                reward = float(np.exp(-mpjpe / self.sigma))
                rewards.append(reward)

            except Exception as e:
                logger.warning(f"GVHMR failed for video_id={vid_id}: {e}, reward=0.0")
                rewards.append(0.0)

        return RewardModelOutput(
            rewards=torch.tensor(rewards, device=self.device, dtype=torch.float32)
        )
