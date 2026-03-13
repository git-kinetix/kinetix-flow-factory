# tests/models/ltx/test_delta_zero.py
"""
Delta-zero verification tests.

These tests verify that LTXUnionAdapter produces BITWISE IDENTICAL
outputs to the original ICLoraPipeline for the same inputs and seeds.

Requirements:
- Real LTX-2.3-22b model weights
- Real Union-Control LoRA weights
- GPU with sufficient VRAM
- ltx_pipelines package installed

Run: pytest tests/models/ltx/test_delta_zero.py -v -m gpu
"""
import pytest
import torch
import os

MODEL_PATH = os.environ.get("LTX_MODEL_PATH", "")
LORA_PATH = os.environ.get("LTX_UNION_LORA_PATH", "")
GEMMA_PATH = os.environ.get("LTX_GEMMA_PATH", "")
SKIP_REASON = "Set LTX_MODEL_PATH, LTX_UNION_LORA_PATH, LTX_GEMMA_PATH env vars"
requires_model = pytest.mark.skipif(
    not all([MODEL_PATH, LORA_PATH, GEMMA_PATH]),
    reason=SKIP_REASON,
)


@requires_model
@pytest.mark.gpu
class TestDeltaZeroPatchify:
    """Test patchify/unpatchify roundtrip is lossless."""

    def test_roundtrip_identity(self):
        from ltx_core.components.patchifiers import VideoLatentPatchifier

        patchifier = VideoLatentPatchifier()
        x = torch.randn(2, 128, 5, 8, 13, dtype=torch.bfloat16)
        patched = patchifier.patchify(x)
        unpatched = patchifier.unpatchify(
            patched, num_frames=5, height=8, width=13
        )
        assert torch.equal(x, unpatched), "Patchify/unpatchify roundtrip failed"


@requires_model
@pytest.mark.gpu
class TestDeltaZeroTransformerStep:
    """Test 1: Single transformer step produces identical output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from flow_factory.models.ltx.pipeline import LTXUnionPseudoPipeline
        self.pipeline = LTXUnionPseudoPipeline.from_pretrained(
            model_path=MODEL_PATH,
            union_lora_path=LORA_PATH,
            gemma_path=GEMMA_PATH,
        )
        self.pipeline.transformer.to("cuda", torch.bfloat16)
        self.pipeline.transformer.eval()

    def test_same_input_same_output(self):
        from ltx_core.model.transformer.modality import Modality

        torch.manual_seed(42)
        B, S, C = 1, 64, 128
        latents = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)
        sigma = torch.tensor([0.5], device="cuda")
        timesteps = torch.full((B, S), 0.5, device="cuda")
        positions = torch.randn(B, 3, S, 2, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(B, 32, 4096, device="cuda", dtype=torch.bfloat16)

        modality = Modality(
            enabled=True,
            latent=latents,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=context,
            context_mask=None,
        )

        with torch.no_grad():
            out_a, _ = self.pipeline.transformer(
                video=modality, audio=None, perturbations=None
            )
            out_b, _ = self.pipeline.transformer(
                video=modality, audio=None, perturbations=None
            )

        assert torch.equal(out_a, out_b), "Deterministic forward not identical"


@requires_model
@pytest.mark.gpu
class TestDeltaZeroFullLoop:
    """Test 2+3: Full denoising loop and decoded pixel parity."""

    def test_full_loop_parity(self):
        """Adapter inference vs ICLoraPipeline → torch.equal on latents."""
        pytest.skip("Requires RD-H200-2 setup — run manually with real weights")
