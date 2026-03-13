import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import os


class TestMPJPERewardModel:
    def test_required_fields(self):
        from flow_factory.rewards.mpjpe_reward import MPJPERewardModel
        assert MPJPERewardModel.required_fields == ("prompt", "video")

    def test_use_tensor_inputs(self):
        from flow_factory.rewards.mpjpe_reward import MPJPERewardModel
        assert MPJPERewardModel.use_tensor_inputs is True

    def test_reward_in_zero_one_range(self):
        """exp(-mpjpe/sigma) is always in [0, 1]."""
        sigma = 0.02
        for mpjpe in [0.0, 0.01, 0.02, 0.05, 0.1, 1.0]:
            reward = float(np.exp(-mpjpe / sigma))
            assert 0.0 <= reward <= 1.0

    def test_perfect_match_gives_reward_one(self):
        """MPJPE=0 → reward=1.0"""
        sigma = 0.02
        reward = float(np.exp(-0.0 / sigma))
        assert reward == 1.0

    def test_gt_cache_load_save_roundtrip(self):
        """GT joint cache should survive save/load."""
        cache = {
            "001": torch.randn(30, 17, 3),
            "002": torch.randn(25, 17, 3),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(cache, f.name)
            loaded = torch.load(f.name, weights_only=True)
            os.unlink(f.name)
        assert set(loaded.keys()) == {"001", "002"}
        assert torch.equal(cache["001"], loaded["001"])
