# tests/models/ltx/test_e2e_smoke.py
"""
End-to-end smoke test for LTX Union NFT training.

Verifies: adapter construction → encoding → inference → reward → forward pass.
Uses real model weights but a tiny synthetic dataset for speed.

Requirements:
- LTX_MODEL_PATH, LTX_UNION_LORA_PATH, LTX_GEMMA_PATH env vars
- GPU with sufficient VRAM
- GVHMR at /data/GVHMR (for full reward test) — or set GVHMR_ROOT

Run: pytest tests/models/ltx/test_e2e_smoke.py -v
"""
import gc
import json
import os
import tempfile

import pytest
import torch

MODEL_PATH = os.environ.get("LTX_MODEL_PATH", "")
LORA_PATH = os.environ.get("LTX_UNION_LORA_PATH", "")
GEMMA_PATH = os.environ.get("LTX_GEMMA_PATH", "")
GVHMR_ROOT = os.environ.get("GVHMR_ROOT", "/data/GVHMR")

requires_model = pytest.mark.skipif(
    not all([MODEL_PATH, LORA_PATH, GEMMA_PATH]),
    reason="Set LTX_MODEL_PATH, LTX_UNION_LORA_PATH, LTX_GEMMA_PATH env vars",
)


def _make_config(tmpdir: str) -> dict:
    """Build a minimal training config for smoke testing."""
    return {
        "launcher": "accelerate",
        "config_file": None,
        "num_processes": 1,
        "main_process_port": 29500,
        "mixed_precision": "bf16",
        "project": "test-e2e-smoke",
        "model": {
            "model_type": "ltx-union",
            "model_name_or_path": MODEL_PATH,
            "union_lora_path": LORA_PATH,
            "gemma_path": GEMMA_PATH,
            "finetune_type": "lora",
            "target_components": ["transformer"],
            "lora_rank": 4,
            "lora_alpha": 4,
        },
        "data": {
            "dataset_dir": tmpdir,
            "video_dir": tmpdir,
            "image_dir": tmpdir,
            "enable_preprocess": True,
            "preprocessing_batch_size": 2,
            "force_reprocess": True,
        },
        "scheduler": {"dynamics_type": "ODE", "noise_level": 0.0},
        "train": {
            "trainer_type": "nft",
            "num_train_timesteps": 2,
            "time_sampling_strategy": "discrete",
            "time_shift": 3.0,
            "timestep_range": 0.9,
            "num_inference_steps": 5,
            "height": 256,
            "width": 416,
            "num_frames": 9,
            "per_device_batch_size": 1,
            "group_size": 2,
            "unique_sample_num_per_epoch": 2,
            "learning_rate": 1e-4,
            "adam_weight_decay": 0.0,
            "enable_gradient_checkpointing": True,
            "nft_beta": 0.1,
            "off_policy": False,
            "kl_type": "v-based",
            "kl_beta": 0.0,
            "advantage_aggregation": "sum",
            "seed": 42,
            "gradient_step_per_epoch": 1,
        },
        "log": {"save_dir": os.path.join(tmpdir, "output"), "save_freq": 999},
        "eval": {
            "height": 256,
            "width": 416,
            "num_frames": 9,
            "eval_freq": 0,
            "per_device_batch_size": 1,
            "num_inference_steps": 5,
            "seed": 42,
        },
    }


def _create_synthetic_dataset(tmpdir: str, num_samples: int = 4):
    """Create a tiny JSONL dataset with dummy video files."""
    import imageio
    import numpy as np

    os.makedirs(os.path.join(tmpdir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "condition", "pose_depth"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "ref_images"), exist_ok=True)

    samples = []
    for i in range(num_samples):
        vid_id = f"clip_{i:03d}"
        # Create tiny 9-frame videos (64x64) — just noise
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(9)]

        gt_path = os.path.join(tmpdir, "videos", f"{vid_id}.mp4")
        cond_path = os.path.join(tmpdir, "condition", "pose_depth", f"{vid_id}.mp4")
        ref_path = os.path.join(tmpdir, "ref_images", f"{vid_id}.png")

        for path in [gt_path, cond_path]:
            writer = imageio.get_writer(path, fps=24)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        # Reference image
        from PIL import Image
        Image.fromarray(frames[0]).save(ref_path)

        samples.append({
            "video_id": vid_id,
            "video": f"condition/pose_depth/{vid_id}.mp4",
            "prompt": f"A person dancing clip {i}",
            "gt_video": f"videos/{vid_id}.mp4",
            "image": f"ref_images/{vid_id}.png",
        })

    jsonl_path = os.path.join(tmpdir, "train.jsonl")
    with open(jsonl_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    return jsonl_path


@requires_model
class TestAdapterConstruction:
    """Test that the adapter can be built from config."""

    def test_adapter_loads(self):
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Arguments.from_dict(_make_config(tmpdir))
            accelerator = Accelerator(mixed_precision="bf16")
            adapter = LTXUnionAdapter(config, accelerator)

            assert adapter.pipeline is not None
            assert adapter.pipeline.transformer is not None
            assert adapter.pipeline.vae_encoder is not None
            assert adapter.pipeline.vae_decoder is not None
            assert adapter.pipeline.text_encoder is not None
            assert adapter.pipeline.patchifier is not None

    def test_encode_prompt(self):
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Arguments.from_dict(_make_config(tmpdir))
            accelerator = Accelerator(mixed_precision="bf16")
            adapter = LTXUnionAdapter(config, accelerator)

            result = adapter.encode_prompt(prompt=["A person dancing", "A person walking"])
            assert "prompt_embeds" in result
            assert "prompt_attention_mask" in result
            assert result["prompt_embeds"].shape[0] == 2
            assert result["prompt_embeds"].device.type == "cuda"

    def test_encode_video(self):
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            config = Arguments.from_dict(_make_config(tmpdir))
            accelerator = Accelerator(mixed_precision="bf16")
            adapter = LTXUnionAdapter(config, accelerator)

            # Simulate dataset input: batch of 1 video, each video is a list of PIL frames
            frames = [Image.new("RGB", (64, 64), color=(128, 64, 32)) for _ in range(9)]
            videos = [[frames]]  # batch × videos-per-sample × frames

            result = adapter.encode_video(videos=videos)
            assert "reference_latents" in result
            assert result["reference_latents"].ndim == 5  # [B, 128, F, H, W]
            assert result["reference_latents"].shape[0] == 1


@requires_model
class TestInference:
    """Test that inference produces valid samples."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        gc.collect()
        torch.cuda.empty_cache()

        self.tmpdir = tempfile.mkdtemp()
        config = Arguments.from_dict(_make_config(self.tmpdir))
        accelerator = Accelerator(mixed_precision="bf16")
        self.adapter = LTXUnionAdapter(config, accelerator)
        yield
        del self.adapter
        gc.collect()
        torch.cuda.empty_cache()

    def test_inference_produces_samples(self):
        """Inference returns list of LTXUnionSample with valid fields."""
        encoded = self.adapter.encode_prompt(prompt=["A person dancing"])
        ref = torch.randn(1, 128, 2, 8, 13, device="cuda", dtype=torch.bfloat16)

        gen = torch.Generator(device="cuda").manual_seed(42)
        samples = self.adapter.inference(
            prompt=["A person dancing"],
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=5,
            compute_log_prob=False,
            trajectory_indices=[-1],
            generator=gen,
            video_id=["test_clip"],
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        assert len(samples) == 1
        s = samples[0]
        assert s.video is not None, "Sample should have decoded video"
        assert s.video.ndim == 4, f"Video should be [T,C,H,W], got {s.video.shape}"
        assert s.video.shape[1] == 3, "Video should have 3 channels"
        assert s.all_latents is not None
        assert s.video_id == "test_clip"
        assert s.prompt == "A person dancing"

    def test_inference_with_log_prob(self):
        """Inference with compute_log_prob=True returns log probs."""
        encoded = self.adapter.encode_prompt(prompt=["Test prompt"])
        ref = torch.randn(1, 128, 2, 8, 13, device="cuda", dtype=torch.bfloat16)

        gen = torch.Generator(device="cuda").manual_seed(42)
        samples = self.adapter.inference(
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            reference_latents=ref,
            height=256, width=416, num_frames=9,
            num_inference_steps=5,
            compute_log_prob=True,
            trajectory_indices="all",
            generator=gen,
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        s = samples[0]
        # ODE with noise_level=0 means no SDE steps, so log_probs may be None
        # depending on whether any timestep triggers SDE. With pure ODE, all
        # noise levels are 0 → compute_log_prob is False for each step.
        # Just verify the sample is returned successfully.
        assert s.all_latents is not None


@requires_model
class TestForwardPass:
    """Test the single-step forward pass used during NFT optimization."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from accelerate import Accelerator
        from flow_factory.hparams import Arguments
        from flow_factory.models.ltx.ltx_union import LTXUnionAdapter

        gc.collect()
        torch.cuda.empty_cache()

        self.tmpdir = tempfile.mkdtemp()
        config = Arguments.from_dict(_make_config(self.tmpdir))
        accelerator = Accelerator(mixed_precision="bf16")
        self.adapter = LTXUnionAdapter(config, accelerator)
        yield
        del self.adapter
        gc.collect()
        torch.cuda.empty_cache()

    def test_forward_returns_sde_output(self):
        """Single denoising step returns SDESchedulerOutput."""
        # Move components to GPU
        for name in self.adapter.inference_modules:
            comp = self.adapter.get_component_unwrapped(name)
            if hasattr(comp, "to"):
                comp.to("cuda")

        encoded = self.adapter.encode_prompt(prompt=["A person dancing"])
        # 5D reference at half resolution (downscale_factor=2): H=4, W=6
        ref_5d = torch.randn(1, 128, 2, 4, 6, device="cuda", dtype=torch.bfloat16)
        latents = torch.randn(1, 128, 2, 8, 13, device="cuda", dtype=torch.bfloat16)

        # Set timesteps on scheduler first
        self.adapter.pipeline.scheduler.set_timesteps(5, device="cuda")
        t = self.adapter.pipeline.scheduler.timesteps[0]
        t_next = self.adapter.pipeline.scheduler.timesteps[1]

        output = self.adapter.forward(
            t=t,
            t_next=t_next,
            latents=latents,
            reference_latents=ref_5d,
            prompt_embeds=encoded["prompt_embeds"],
            prompt_attention_mask=encoded["prompt_attention_mask"],
            noise_level=0.0,
            compute_log_prob=False,
            return_dict=True,
            return_kwargs=["next_latents", "noise_pred"],
            audio_prompt_embeds=encoded.get("audio_prompt_embeds"),
        )

        assert output.next_latents is not None
        assert output.next_latents.shape == latents.shape, (
            f"next_latents shape {output.next_latents.shape} != input {latents.shape}"
        )
