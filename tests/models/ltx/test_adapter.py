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
