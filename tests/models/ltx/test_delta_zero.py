# tests/models/ltx/test_delta_zero.py
"""
Delta-zero verification tests for LTX Union adapter.

Test 1: Patchify/unpatchify roundtrip is bitwise lossless.
Test 2: Single transformer step is deterministic (same input → same output).
Test 3: Full inference loop is deterministic (same seed → same latents).
Test 4: Decoded pixel video is deterministic end-to-end.

Requirements:
- ltx_core package installed
- For Test 2+: LTX_MODEL_PATH, LTX_UNION_LORA_PATH, LTX_GEMMA_PATH env vars
- GPU with sufficient VRAM

Run: pytest tests/models/ltx/test_delta_zero.py -v
"""
import pytest
import torch
import os
import tempfile


def _ltx_core_available():
    try:
        import ltx_core  # noqa: F401
        return True
    except ImportError:
        return False


MODEL_PATH = os.environ.get("LTX_MODEL_PATH", "")
LORA_PATH = os.environ.get("LTX_UNION_LORA_PATH", "")
GEMMA_PATH = os.environ.get("LTX_GEMMA_PATH", "")

requires_ltx_core = pytest.mark.skipif(
    not _ltx_core_available(),
    reason="ltx_core package not installed",
)
requires_model = pytest.mark.skipif(
    not all([MODEL_PATH, LORA_PATH, GEMMA_PATH]),
    reason="Set LTX_MODEL_PATH, LTX_UNION_LORA_PATH, LTX_GEMMA_PATH env vars",
)


class TestDeltaZeroPatchify:
    """Test patchify/unpatchify roundtrip is lossless."""

    @pytest.mark.skipif(not _ltx_core_available(), reason="ltx_core not installed")
    def test_roundtrip_identity(self):
        from ltx_core.components.patchifiers import VideoLatentPatchifier

        patchifier = VideoLatentPatchifier(patch_size=1)
        x = torch.randn(2, 128, 5, 8, 13, dtype=torch.bfloat16)
        patched = patchifier.patchify(x)
        from ltx_core.types import VideoLatentShape
        shape = VideoLatentShape(batch=2, channels=128, frames=5, height=8, width=13)
        unpatched = patchifier.unpatchify(patched, shape)
        assert torch.equal(x, unpatched), "Patchify/unpatchify roundtrip not bitwise identical"

    @pytest.mark.skipif(not _ltx_core_available(), reason="ltx_core not installed")
    def test_different_shapes(self):
        """Roundtrip works for multiple latent resolutions."""
        from ltx_core.components.patchifiers import VideoLatentPatchifier
        from ltx_core.types import VideoLatentShape

        patchifier = VideoLatentPatchifier(patch_size=1)
        for F, H, W in [(5, 8, 13), (1, 4, 7), (9, 16, 26)]:
            x = torch.randn(1, 128, F, H, W, dtype=torch.bfloat16)
            patched = patchifier.patchify(x)
            shape = VideoLatentShape(batch=1, channels=128, frames=F, height=H, width=W)
            unpatched = patchifier.unpatchify(patched, shape)
            assert torch.equal(x, unpatched), f"Roundtrip failed for shape ({F},{H},{W})"


@requires_model
class TestDeltaZeroTransformerStep:
    """Test single transformer step produces identical output."""

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
        yield
        del self.pipeline

    def test_same_input_same_output(self):
        from ltx_core.model.transformer.modality import Modality

        torch.manual_seed(42)
        B, S, C = 1, 64, 128
        latents = torch.randn(B, S, C, device="cuda", dtype=torch.bfloat16)
        sigma = torch.tensor([0.5], device="cuda")
        timesteps = torch.full((B, S), 0.5, device="cuda")
        positions = torch.randn(B, 3, S, 2, device="cuda", dtype=torch.bfloat16)
        # Gemma-3 hidden_size=3840 — transformer's text_projection: Linear(3840→4096)
        context = torch.randn(B, 32, 3840, device="cuda", dtype=torch.bfloat16)

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
class TestDeltaZeroFullLoop:
    """Test full denoising loop and decoded pixel parity."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Construct the adapter the same way the training pipeline does."""
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments

        # Build a minimal config with real model paths
        config_dict = {
            "launcher": "accelerate",
            "config_file": None,
            "num_processes": 1,
            "main_process_port": 29500,
            "mixed_precision": "bf16",
            "project": "test-delta-zero",
            "model": {
                "model_type": "ltx-union",
                "model_name_or_path": MODEL_PATH,
                "union_lora_path": LORA_PATH,
                "gemma_path": GEMMA_PATH,
                "finetune_type": "full",
            },
            "train": {
                "trainer_type": "nft",
                "num_inference_steps": 10,
                "height": 256,
                "width": 416,
                "num_frames": 9,
                "per_device_batch_size": 1,
                "group_size": 1,
                "unique_sample_num_per_epoch": 1,
                "num_train_timesteps": 2,
                "time_sampling_strategy": "discrete",
                "time_shift": 3.0,
                "timestep_range": 0.9,
            },
            "scheduler": {"dynamics_type": "ODE", "noise_level": 0.0},
            "log": {"save_dir": tempfile.mkdtemp(), "save_freq": 999},
        }
        config = Arguments.from_dict(config_dict)
        accelerator = Accelerator(mixed_precision="bf16")

        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        self.adapter = LTXUnionAdapter(config, accelerator)
        self.config = config
        yield
        del self.adapter

    def test_full_loop_deterministic(self):
        """Same seed → bitwise identical latents."""
        # Encode a test prompt
        encoded = self.adapter.encode_prompt(prompt=["A person dancing"])
        # Create dummy reference latents (small)
        ref = torch.randn(1, 128, 2, 8, 13, device="cuda", dtype=torch.bfloat16)

        # Run inference twice with same seed
        gen_a = torch.Generator(device="cuda").manual_seed(123)
        samples_a = self.adapter.inference(
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=10,
            compute_log_prob=False,
            generator=gen_a,
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        gen_b = torch.Generator(device="cuda").manual_seed(123)
        samples_b = self.adapter.inference(
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=10,
            compute_log_prob=False,
            generator=gen_b,
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        # Compare final latents
        lat_a = samples_a[0].all_latents
        lat_b = samples_b[0].all_latents
        assert torch.equal(lat_a, lat_b), (
            f"Full loop not deterministic! Max diff: {(lat_a - lat_b).abs().max().item()}"
        )

    def test_decoded_pixels_deterministic(self):
        """Same seed → bitwise identical decoded video pixels."""
        encoded = self.adapter.encode_prompt(prompt=["A person walking"])
        ref = torch.randn(1, 128, 2, 8, 13, device="cuda", dtype=torch.bfloat16)

        gen_a = torch.Generator(device="cuda").manual_seed(456)
        samples_a = self.adapter.inference(
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=10,
            compute_log_prob=False,
            generator=gen_a,
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        gen_b = torch.Generator(device="cuda").manual_seed(456)
        samples_b = self.adapter.inference(
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=10,
            compute_log_prob=False,
            generator=gen_b,
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        vid_a = samples_a[0].video  # [T, C, H, W]
        vid_b = samples_b[0].video
        assert torch.equal(vid_a, vid_b), (
            f"Decoded video not deterministic! Max diff: {(vid_a - vid_b).abs().max().item()}"
        )
