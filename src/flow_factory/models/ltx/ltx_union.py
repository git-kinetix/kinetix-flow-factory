"""
LTX Union Adapter for Flow-Factory.

Wraps the LTX-2.3-22b model with merged Union-Control LoRA into
Flow-Factory's BaseAdapter interface for DiffusionNFT fine-tuning.

Follows the 7-step new model guide:
1. Sample dataclass (LTXUnionSample)
2. Adapter class (LTXUnionAdapter)
3. Module properties
4. Encoding methods
5. inference()
6. forward()
7. Registry entry
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union,
)

import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator
from peft import PeftModel

from ..abc import BaseAdapter
from ...samples import VideoConditionSample
from ...hparams import Arguments
from ...scheduler import SDESchedulerOutput
from ...utils.logger_utils import setup_logger
from .pipeline import LTXUnionPseudoPipeline
from .scheduler_bridge import LTXSDEScheduler

logger = setup_logger(__name__)


@dataclass
class LTXUnionSample(VideoConditionSample):
    """Sample for LTX Union IC-LoRA video generation."""

    _shared_fields: ClassVar[frozenset] = frozenset({})

    ref_seq_len: Optional[int] = None
    video_id: Optional[str] = None
    reference_latents: Optional[torch.Tensor] = None


class LTXUnionAdapter(BaseAdapter):
    """Flow-Factory adapter for LTX-2.3-22b with Union-Control LoRA."""

    def __init__(self, config: Arguments, accelerator: Accelerator):
        super().__init__(config, accelerator)
        self.pipeline: LTXUnionPseudoPipeline
        self.reference_downscale_factor: int = self.pipeline.config.get(
            "reference_downscale_factor", 1
        )

    # ======================== Loading ========================

    def load_pipeline(self) -> LTXUnionPseudoPipeline:
        return LTXUnionPseudoPipeline.from_pretrained(
            model_path=self.model_args.model_name_or_path,
            union_lora_path=getattr(self.model_args, "union_lora_path", None),
            gemma_path=getattr(self.model_args, "gemma_path", None),
            low_cpu_mem_usage=False,
        )

    def load_scheduler(self) -> LTXSDEScheduler:
        """Override: bypass diffusers scheduler registry."""
        return LTXSDEScheduler(
            num_inference_steps=getattr(
                self.config.training_args, "num_inference_steps", 50
            ),
            dynamics_type=getattr(
                self.config.scheduler_args, "dynamics_type", "ODE"
            ),
            noise_level=getattr(
                self.config.scheduler_args, "noise_level", 0.0
            ),
        )

    # ======================== BaseAdapter Hook Overrides ========================

    def _mix_precision(self):
        """Override: cast vae_decoder and embeddings_processor too."""
        super()._mix_precision()
        inference_dtype = self._inference_dtype
        self.pipeline.vae_decoder.to(dtype=inference_dtype)
        self.pipeline.embeddings_processor.to(dtype=inference_dtype)

    def enable_gradient_checkpointing(self):
        """Override: LTXModel uses set_gradient_checkpointing(enable=True)."""
        if hasattr(self.pipeline.transformer, "set_gradient_checkpointing"):
            self.pipeline.transformer.set_gradient_checkpointing(enable=True)
        elif hasattr(self.pipeline.transformer, "gradient_checkpointing_enable"):
            self.pipeline.transformer.gradient_checkpointing_enable()
        else:
            logger.warning(
                "LTXModel has no gradient checkpointing method; skipping"
            )

    def apply_lora(
        self,
        target_modules: Union[str, List[str]],
        components: Union[str, List[str]] = ["transformer"],
        **kwargs,
    ) -> Union[PeftModel, Dict[str, PeftModel]]:
        return super().apply_lora(
            target_modules=target_modules, components=components, **kwargs
        )

    # ======================== Module Properties ========================

    @property
    def default_target_modules(self) -> List[str]:
        return [
            "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
            "ff.net.0.proj", "ff.net.2",
        ]

    @property
    def preprocessing_modules(self) -> List[str]:
        return ["text_encoder", "embeddings_processor", "vae_encoder"]

    @property
    def inference_modules(self) -> List[str]:
        return [
            "transformer", "vae_encoder", "vae_decoder",
            "patchifier", "embeddings_processor",
        ]

    @property
    def text_encoder_names(self) -> List[str]:
        return ["text_encoder"]

    # ======================== Encoding ========================

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Encode text prompts via Gemma blocks 1+2 (precompute path)."""
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_encoder = self.get_component_unwrapped("text_encoder")
        video_features, audio_features, attention_mask = text_encoder.precompute(
            prompt, padding_side="left"
        )

        tokenizer = text_encoder.tokenizer
        tokenized = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "prompt_ids": tokenized.input_ids.to(device),
            "prompt_embeds": video_features.to(device=device),
            "prompt_attention_mask": attention_mask.to(device),
            "_audio_prompt_embeds": audio_features.to(device=device),
        }

    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        **kwargs,
    ) -> Optional[Dict[str, Union[List[Any], torch.Tensor]]]:
        """Not used for LTX Union. Conditioning via encode_video (IC-LoRA ref)."""
        return None

    def encode_video(
        self,
        videos: Union[torch.Tensor, List[Image.Image], List[List[Image.Image]]],
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Encode pose_depth condition video into reference latents."""
        import torchvision.transforms.functional as TF

        vae_encoder = self.get_component_unwrapped("vae_encoder")
        device = self.device

        # Handle pre-batched tensor input (B, C, T, H, W)
        if isinstance(videos, torch.Tensor):
            enc_dtype = getattr(vae_encoder, "dtype", None)
            video_tensor = videos.to(device=device)
            if isinstance(enc_dtype, torch.dtype):
                video_tensor = video_tensor.to(dtype=enc_dtype)
            latents = vae_encoder(video_tensor)
            return {"reference_latents": latents}

        if isinstance(videos[0], Image.Image):
            videos = [videos]

        ref_latents_list = []
        for video_frames in videos:
            tensors = []
            for frame in video_frames:
                t = TF.to_tensor(frame)
                tensors.append(t)
            video_tensor = torch.stack(tensors, dim=1).unsqueeze(0)
            enc_dtype = getattr(vae_encoder, "dtype", None)
            video_tensor = video_tensor.to(device=device)
            if isinstance(enc_dtype, torch.dtype):
                video_tensor = video_tensor.to(dtype=enc_dtype)
            latent = vae_encoder(video_tensor)
            ref_latents_list.append(latent.squeeze(0))

        reference_latents = torch.stack(ref_latents_list, dim=0)
        return {"reference_latents": reference_latents}

    def preprocess_func(
        self,
        prompt: List[str],
        videos: Optional[Any] = None,
        images: Optional[Any] = None,
        video_id: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Union[List[Any], torch.Tensor]]:
        """Custom preprocessing: encode prompt + reference video + pass video_id."""
        results = {}

        if prompt is not None:
            results.update(self.encode_prompt(prompt=prompt, **kwargs))

        if videos is not None:
            results.update(self.encode_video(videos=videos, **kwargs))

        if video_id is not None:
            results["video_id"] = video_id

        return results

    # ======================== Decoding ========================

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: Literal["pt", "pil", "np"] = "pil",
        **kwargs,
    ) -> Union[torch.Tensor, List[Image.Image]]:
        """Decode latent video to pixel space.

        Args:
            latents: [B, 128, F, H, W] unpatchified latents.
            output_type: Output format.
        """
        from ltx_core.model.video_vae import decode_video as vae_decode_video

        vae_decoder = self.get_component_unwrapped("vae_decoder")
        videos = []
        for i in range(latents.shape[0]):
            decoded = vae_decode_video(
                latents[i : i + 1], vae_decoder, tiling_config=None, generator=None
            )
            videos.append(decoded)

        if output_type == "pt":
            return torch.cat(videos, dim=0)
        else:
            return videos

    # ======================== Forward (single denoising step) ========================

    def forward(
        self,
        # Timestep info
        t: torch.Tensor,
        t_next: Optional[torch.Tensor] = None,
        # Latent state (5D unpatchified from NFT trainer)
        latents: Optional[torch.Tensor] = None,
        next_latents: Optional[torch.Tensor] = None,
        # Reference latents (3D patchified)
        reference_latents: Optional[torch.Tensor] = None,
        ref_seq_len: Optional[int] = None,
        # Conditioning
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        # Control flags
        noise_level: Optional[float] = None,
        compute_log_prob: bool = True,
        return_kwargs: List[str] = [
            "noise_pred", "next_latents", "log_prob",
            "next_latents_mean", "std_dev_t", "dt",
        ],
        **kwargs,
    ) -> SDESchedulerOutput:
        """Single denoising step with IC-LoRA reference concatenation.

        The LTX transformer outputs velocity v. The NFT trainer treats
        this as noise_pred — the naming is cosmetic.

        Flow:
        1. Patchify 5D target latents → 3D tokens
        2. Concatenate [ref_latents | target_latents]
        3. Build per-token timesteps + positions + Modality
        4. Transformer forward
        5. Extract target prediction, unpatchify
        6. Scheduler step → SDESchedulerOutput
        """
        from ltx_core.model.transformer.modality import Modality

        batch_size = latents.shape[0]
        device = latents.device
        dtype = latents.dtype

        patchifier = self.get_component_unwrapped("patchifier")
        transformer = self.get_component("transformer")  # accelerator-wrapped

        # 1. Patchify target latents: [B, 128, F, H, W] → [B, seq_target, 128]
        target_latents_3d = patchifier.patchify(latents)
        seq_target = target_latents_3d.shape[1]

        next_latents_3d = None
        if next_latents is not None:
            next_latents_3d = patchifier.patchify(next_latents)

        # 2. Concatenate reference + target
        assert reference_latents is not None, "reference_latents required for IC-LoRA"
        seq_ref = reference_latents.shape[1]
        combined = torch.cat([reference_latents, target_latents_3d], dim=1)

        # 3. Build per-token timesteps
        sigma = t / 1000.0
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        sigma_batch = sigma.view(-1)
        if sigma_batch.shape[0] == 1:
            sigma_batch = sigma_batch.expand(batch_size)

        ref_ts = torch.zeros(batch_size, seq_ref, device=device, dtype=torch.float32)
        target_ts = sigma_batch.unsqueeze(1).expand(batch_size, seq_target)
        per_token_timesteps = torch.cat([ref_ts, target_ts], dim=1)

        # 4. Build positions
        _, _, F_lat, H_lat, W_lat = latents.shape
        ref_H = H_lat // self.reference_downscale_factor
        ref_W = W_lat // self.reference_downscale_factor

        target_positions = patchifier.get_patch_grid_bounds(
            num_frames=F_lat, height=H_lat, width=W_lat,
            batch_size=batch_size, device=device, dtype=dtype,
        )

        ref_positions = patchifier.get_patch_grid_bounds(
            num_frames=F_lat, height=ref_H, width=ref_W,
            batch_size=batch_size, device=device, dtype=dtype,
        )

        if self.reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= self.reference_downscale_factor
            ref_positions[:, 2, ...] *= self.reference_downscale_factor

        positions = torch.cat([ref_positions, target_positions], dim=2)

        # 5. Apply embeddings processor connectors (block 3)
        embeddings_processor = self.get_component_unwrapped("embeddings_processor")
        additive_mask = None
        if prompt_attention_mask is not None:
            additive_mask = prompt_attention_mask
        video_context, _, _ = embeddings_processor.create_embeddings(
            prompt_embeds,
            kwargs.get("_audio_prompt_embeds", prompt_embeds),
            additive_mask,
        )

        # 6. Build Modality
        modality = Modality(
            enabled=True,
            latent=combined,
            sigma=sigma_batch,
            timesteps=per_token_timesteps,
            positions=positions,
            context=video_context,
            context_mask=prompt_attention_mask,
        )

        # 7. Transformer forward
        video_pred, _ = transformer(
            video=modality, audio=None, perturbations=None
        )

        # 8. Extract target prediction and unpatchify
        target_v_pred_3d = video_pred[:, seq_ref:]
        v_pred_5d = patchifier.unpatchify(
            target_v_pred_3d,
            num_frames=F_lat, height=H_lat, width=W_lat,
        )

        # 9. Scheduler step (operates in 5D)
        output = self.pipeline.scheduler.step(
            noise_pred=v_pred_5d,
            timestep=t,
            latents=latents,
            next_latents=next_latents,
            timestep_next=t_next,
            noise_level=noise_level,
            compute_log_prob=compute_log_prob,
            return_dict=True,
            return_kwargs=return_kwargs,
        )

        return output

    def inference(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 7")
