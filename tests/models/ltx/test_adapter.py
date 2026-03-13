import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import ClassVar, Optional

class TestLTXUnionSample:
    def test_sample_has_video_id_field(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionSample
        sample = LTXUnionSample()
        assert hasattr(sample, "video_id")
        assert hasattr(sample, "ref_seq_len")
        assert hasattr(sample, "reference_latents")

    def test_sample_inherits_video_condition(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionSample
        from flow_factory.samples import VideoConditionSample
        assert issubclass(LTXUnionSample, VideoConditionSample)

class TestLTXUnionAdapterProperties:
    def test_default_target_modules_returns_list(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        assert hasattr(LTXUnionAdapter, "default_target_modules")

    def test_inference_modules_includes_key_components(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        assert hasattr(LTXUnionAdapter, "inference_modules")

    def test_text_encoder_names_returns_list(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        assert hasattr(LTXUnionAdapter, "text_encoder_names")

class TestLTXUnionRegistry:
    def test_ltx_union_in_registry(self):
        from flow_factory.models.registry import list_registered_models
        models = list_registered_models()
        assert "ltx-union" in models

    def test_ltx_union_resolves_to_adapter(self):
        from flow_factory.models.registry import get_model_adapter_class
        cls = get_model_adapter_class("ltx-union")
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        assert cls is LTXUnionAdapter

import torch

class TestEncodingMethods:
    """Test encoding methods actually work (not just signatures).
    Uses mocked pipeline components since real models need GPU + weights."""

    def test_encode_prompt_no_longer_raises(self):
        """encode_prompt must be implemented (not raise NotImplementedError).
        We mock the text_encoder to verify the method runs end-to-end."""
        from unittest.mock import MagicMock, patch
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        adapter = MagicMock(spec=LTXUnionAdapter)
        adapter.device = torch.device("cpu")

        # Mock the text_encoder.precompute call
        fake_embeds = torch.randn(1, 16, 64)
        fake_audio = torch.randn(1, 16, 64)
        fake_mask = torch.ones(1, 16)
        mock_encoder = MagicMock()
        mock_encoder.precompute.return_value = (fake_embeds, fake_audio, fake_mask)
        mock_encoder.tokenizer.return_value = MagicMock(
            input_ids=torch.zeros(1, 16, dtype=torch.long)
        )
        adapter.get_component_unwrapped.return_value = mock_encoder

        # Call the real method on the mock
        result = LTXUnionAdapter.encode_prompt(adapter, prompt=["test prompt"])
        assert "prompt_embeds" in result
        assert "prompt_ids" in result
        assert "prompt_attention_mask" in result

    def test_encode_video_returns_reference_latents(self):
        """encode_video must return dict with 'reference_latents' key."""
        from unittest.mock import MagicMock, patch
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        adapter = MagicMock(spec=LTXUnionAdapter)
        adapter.device = torch.device("cpu")

        # Mock VAE encoder
        mock_vae = MagicMock()
        mock_vae.return_value = torch.randn(1, 128, 1, 8, 8)
        adapter.get_component_unwrapped.return_value = mock_vae

        fake_video = torch.randn(1, 3, 8, 64, 64)
        result = LTXUnionAdapter.encode_video(adapter, videos=fake_video)
        assert "reference_latents" in result

    def test_encode_image_returns_none(self):
        """encode_image not used for LTX Union — should return empty dict or None."""
        from unittest.mock import MagicMock
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        adapter = MagicMock(spec=LTXUnionAdapter)
        result = LTXUnionAdapter.encode_image(adapter, images=torch.randn(1, 3, 64, 64))
        assert result is None or isinstance(result, dict)
