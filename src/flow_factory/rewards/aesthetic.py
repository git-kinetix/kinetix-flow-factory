"""
Aesthetic Score Reward Model.

Uses CLIP ViT-L/14 embeddings with LAION's aesthetic predictor MLP head.
Scores frames from generated videos and returns the mean aesthetic score
normalized to [0, 1] range.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image

from .abc import PointwiseRewardModel, RewardModelOutput
from ..hparams import RewardArguments
from ..utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class _AestheticMLP(nn.Module):
    """Simple MLP head matching LAION aesthetic predictor v2 architecture."""

    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticScoreRewardModel(PointwiseRewardModel):
    """Pointwise reward based on LAION aesthetic score (1-10 scale, normalized to 0-1)."""

    required_fields = ("prompt", "video")
    use_tensor_inputs = False

    def __init__(self, config: RewardArguments, accelerator):
        super().__init__(config, accelerator)
        self._model = None
        self._preprocess = None
        self._mlp = None
        self.num_frames = getattr(config, "num_frames", 4)
        self.clip_model_name = getattr(
            config, "clip_model_name", "ViT-L-14"
        )
        self.clip_pretrained = getattr(
            config, "clip_pretrained", "openai"
        )
        self.aesthetic_mlp_url = getattr(
            config, "aesthetic_mlp_url",
            "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth",
        )

    def _load(self):
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        model = model.to(self.device).eval()

        mlp = _AestheticMLP(input_dim=768)
        state_dict = torch.hub.load_state_dict_from_url(
            self.aesthetic_mlp_url, map_location="cpu"
        )
        mlp.load_state_dict(state_dict)
        mlp = mlp.to(self.device).eval()

        self._model = model
        self._preprocess = preprocess
        self._mlp = mlp
        logger.info("Loaded aesthetic score model (CLIP ViT-L/14 + LAION MLP)")

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        image: Optional[List[Image.Image]] = None,
        video: Optional[List[List[Image.Image]]] = None,
        condition_images: Optional[List] = None,
        condition_videos: Optional[List] = None,
        **kwargs,
    ) -> RewardModelOutput:
        if self._model is None:
            self._load()

        if video is None and image is None:
            return RewardModelOutput(
                rewards=torch.zeros(len(prompt), device=self.device)
            )

        rewards = []
        items = video if video is not None else [[img] for img in image]

        for frames in items:
            # Sample a few frames evenly
            if len(frames) > self.num_frames:
                indices = torch.linspace(0, len(frames) - 1, self.num_frames).long()
                sampled = [frames[i] for i in indices]
            else:
                sampled = list(frames)

            # Preprocess and embed
            tensors = torch.stack([self._preprocess(f) for f in sampled]).to(self.device)
            features = self._model.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)

            # Score each frame and average
            scores = self._mlp(features.float()).squeeze(-1)  # (num_frames,)
            # LAION scores are roughly 1-10; normalize to 0-1
            mean_score = scores.mean().clamp(1.0, 10.0)
            reward = (mean_score - 1.0) / 9.0
            rewards.append(reward.item())

        return RewardModelOutput(
            rewards=torch.tensor(rewards, device=self.device, dtype=torch.float32)
        )
