import pytest
import torch
import torch.nn as nn

class FakeModule(nn.Module):
    """Minimal stand-in for any pipeline component."""
    def __init__(self, name="fake"):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self._name = name

def make_fake_pipeline():
    from flow_factory.models.ltx.pipeline import LTXUnionPseudoPipeline
    return LTXUnionPseudoPipeline(
        transformer=FakeModule("transformer"),
        vae_encoder=FakeModule("vae_encoder"),
        vae_decoder=FakeModule("vae_decoder"),
        text_encoder=FakeModule("text_encoder"),
        embeddings_processor=FakeModule("embeddings_processor"),
        patchifier=FakeModule("patchifier"),
        config={"model_type": "ltx-union"},
    )

class TestLTXUnionPseudoPipeline:
    def test_construction(self):
        pipe = make_fake_pipeline()
        assert pipe.transformer is not None
        assert pipe.vae_encoder is not None
        assert pipe.vae_decoder is not None
        assert pipe.text_encoder is not None
        assert pipe.embeddings_processor is not None
        assert pipe.patchifier is not None
        assert pipe.scheduler is None  # Set later by adapter

    def test_vae_proxy_returns_vae_encoder(self):
        pipe = make_fake_pipeline()
        assert pipe.vae is pipe.vae_encoder

    def test_vae_proxy_dtype_cast_propagates(self):
        """BaseAdapter._mix_precision() calls self.pipeline.vae.to(dtype=...).
        Verify the proxy propagates dtype changes to vae_encoder."""
        pipe = make_fake_pipeline()
        pipe.vae.to(dtype=torch.float16)
        assert pipe.vae_encoder.linear.weight.dtype == torch.float16

    def test_component_attributes_discoverable(self):
        """BaseAdapter scans vars(self.pipeline) for component discovery."""
        pipe = make_fake_pipeline()
        attrs = vars(pipe)
        assert "transformer" in attrs
        assert "text_encoder" in attrs
        assert "vae_encoder" in attrs
        assert "vae_decoder" in attrs
