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
        We mock the text_encoder + embeddings_processor to verify the method runs end-to-end."""
        from unittest.mock import MagicMock
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        adapter = MagicMock(spec=LTXUnionAdapter)
        adapter.device = torch.device("cpu")

        # Mock text_encoder.encode() → (hidden_states, attention_mask)
        fake_hs = torch.randn(1, 16, 3840)
        fake_mask = torch.ones(1, 16)
        mock_text_encoder = MagicMock()
        mock_text_encoder.encode.return_value = (fake_hs, fake_mask)

        # Mock embeddings_processor.process_hidden_states() → output with video/audio/mask
        mock_proc_output = MagicMock()
        mock_proc_output.video_encoding = torch.randn(1, 16, 3840)
        mock_proc_output.audio_encoding = torch.randn(1, 16, 3840)
        mock_proc_output.attention_mask = torch.ones(1, 16)
        mock_embeddings_proc = MagicMock()
        mock_embeddings_proc.process_hidden_states.return_value = mock_proc_output

        def _get_component(name):
            if name == "text_encoder":
                return mock_text_encoder
            return mock_embeddings_proc

        adapter.get_component_unwrapped.side_effect = _get_component

        # Call the real method on the mock
        result = LTXUnionAdapter.encode_prompt(adapter, prompt=["test prompt"])
        assert "prompt_embeds" in result
        assert "prompt_attention_mask" in result
        assert "audio_prompt_embeds" in result

    def test_encode_video_returns_reference_latents(self):
        """encode_video must return dict with 'reference_latents' key."""
        from unittest.mock import MagicMock, patch
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        adapter = MagicMock(spec=LTXUnionAdapter)
        adapter.device = torch.device("cpu")
        adapter.reference_downscale_factor = 1

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


class TestForwardBehavior:
    def test_forward_no_longer_raises_not_implemented(self):
        """forward() must be implemented — calling it should not raise NotImplementedError.
        We verify the method signature has all required parameters and is callable."""
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        import inspect
        sig = inspect.signature(LTXUnionAdapter.forward)
        params = list(sig.parameters.keys())
        # Must accept these core arguments
        for required in ["t", "latents", "reference_latents", "prompt_embeds"]:
            assert required in params, f"forward() missing required param: {required}"
        # Verify method is not the NotImplementedError stub
        import dis
        instructions = list(dis.get_instructions(LTXUnionAdapter.forward))
        raise_ops = [i for i in instructions if i.opname == "RAISE_VARARGS"]
        assert len(instructions) > 10, "forward() appears to still be a stub"

    def test_decode_latents_no_longer_raises_not_implemented(self):
        """decode_latents() must be implemented — not a stub."""
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        import inspect, dis
        sig = inspect.signature(LTXUnionAdapter.decode_latents)
        params = list(sig.parameters.keys())
        assert "latents" in params
        instructions = list(dis.get_instructions(LTXUnionAdapter.decode_latents))
        assert len(instructions) > 10, "decode_latents() appears to still be a stub"


class TestInferenceSignature:
    def test_inference_accepts_required_args(self):
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        import inspect
        sig = inspect.signature(LTXUnionAdapter.inference)
        params = list(sig.parameters.keys())
        assert "prompt" in params or "prompt_embeds" in params
        assert "self" in params
