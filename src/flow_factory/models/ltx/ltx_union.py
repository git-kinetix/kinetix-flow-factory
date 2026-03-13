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
            "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
            "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0",
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
        """Encode text prompts via Gemma text_encoder + embeddings_processor.

        Two-stage encoding:
        1. text_encoder.encode(text) → raw hidden states + attention mask
        2. embeddings_processor.process_hidden_states() → video/audio embeddings
        """
        device = device or self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        text_encoder = self.get_component_unwrapped("text_encoder")
        embeddings_processor = self.get_component_unwrapped("embeddings_processor")

        # Ensure components are on the right device
        text_encoder.to(device)
        embeddings_processor.to(device)

        video_encs, audio_encs, masks = [], [], []
        for text in prompt:
            with torch.no_grad():
                hidden_states, attention_mask = text_encoder.encode(text)
                out = embeddings_processor.process_hidden_states(
                    hidden_states, attention_mask
                )
            video_encs.append(out.video_encoding)
            audio_encs.append(out.audio_encoding)
            masks.append(out.attention_mask)

        prompt_embeds = torch.cat(video_encs, dim=0).to(device=device)
        prompt_attention_mask = torch.cat(masks, dim=0).to(device=device)
        audio_prompt_embeds = (
            torch.cat(audio_encs, dim=0).to(device=device)
            if audio_encs[0] is not None else None
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "audio_prompt_embeds": audio_prompt_embeds,
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

        # Ensure encoder is on the right device
        vae_encoder.to(device)

        # Detect encoder dtype from parameters (nn.Module has no .dtype attr)
        try:
            enc_dtype = next(vae_encoder.parameters()).dtype
        except StopIteration:
            enc_dtype = None

        if isinstance(videos, torch.Tensor):
            video_tensor = videos.to(device=device)
            if isinstance(enc_dtype, torch.dtype):
                video_tensor = video_tensor.to(dtype=enc_dtype)
            latents = vae_encoder(video_tensor)
            return {"reference_latents": latents}

        # Unwrap nested dataset format: List[List[List[PIL]]] → List[List[PIL]]
        # Dataset provides [[frames_sample1], [frames_sample2], ...] (one video per sample)
        if (isinstance(videos, list) and len(videos) > 0
                and isinstance(videos[0], list) and len(videos[0]) > 0
                and isinstance(videos[0][0], list)):
            videos = [v[0] for v in videos]

        if isinstance(videos[0], Image.Image):
            videos = [videos]

        ref_latents_list = []
        for video_frames in videos:
            tensors = []
            for frame in video_frames:
                t = TF.to_tensor(frame)
                tensors.append(t)
            video_tensor = torch.stack(tensors, dim=1).unsqueeze(0)
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
            # C4 fix: decode_video takes 4D [C,F,H,W], yields [F,H,W,C] uint8 chunks
            frame_chunks = list(vae_decode_video(
                latents[i], vae_decoder, tiling_config=None, generator=None
            ))
            decoded = torch.cat(frame_chunks, dim=0)  # [total_F, H, W, C] uint8
            # Convert to [T, C, H, W] float [0,1] — expected by BaseSample/rewards
            decoded = decoded.permute(0, 3, 1, 2).float() / 255.0
            videos.append(decoded)

        if output_type == "pt":
            return torch.stack(videos, dim=0)  # [B, T, C, H, W]
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
        from ltx_core.types import VideoLatentShape, VIDEO_SCALE_FACTORS
        from ltx_core.components.patchifiers import get_pixel_coords
        from ltx_core.text_encoders.gemma import convert_to_additive_mask

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

        # 4. Build positions using VideoLatentShape (C2 fix)
        _, _, F_lat, H_lat, W_lat = latents.shape
        ref_H = H_lat // self.reference_downscale_factor
        ref_W = W_lat // self.reference_downscale_factor

        target_shape = VideoLatentShape(
            batch=batch_size, channels=128, frames=F_lat,
            height=H_lat, width=W_lat,
        )
        target_positions = patchifier.get_patch_grid_bounds(
            target_shape, device=device,
        )

        ref_shape = VideoLatentShape(
            batch=batch_size, channels=128, frames=F_lat,
            height=ref_H, width=ref_W,
        )
        ref_positions = patchifier.get_patch_grid_bounds(
            ref_shape, device=device,
        )

        if self.reference_downscale_factor != 1:
            ref_positions = ref_positions.clone()
            ref_positions[:, 1, ...] *= self.reference_downscale_factor
            ref_positions[:, 2, ...] *= self.reference_downscale_factor

        positions = torch.cat([ref_positions, target_positions], dim=2)

        # Convert latent-space positions to pixel-space (C5 fix)
        positions = get_pixel_coords(positions, VIDEO_SCALE_FACTORS, causal_fix=True)
        # Temporal axis must be in seconds, not frame indices
        fps = getattr(self.pipeline, "fps", 24.0)
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] = positions[:, 0, ...] / fps

        # 5. Apply embeddings processor connectors (block 3)
        # C6 fix: convert binary mask → additive mask for transformer attention
        embeddings_processor = self.get_component_unwrapped("embeddings_processor")
        additive_mask = convert_to_additive_mask(prompt_attention_mask, dtype=dtype)
        video_context, _, _ = embeddings_processor.create_embeddings(
            prompt_embeds,
            kwargs.get("audio_prompt_embeds", prompt_embeds),
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

        # 8. Extract target prediction and unpatchify (C3 fix: use VideoLatentShape)
        target_v_pred_3d = video_pred[:, seq_ref:]
        v_pred_5d = patchifier.unpatchify(target_v_pred_3d, target_shape)

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

    # ======================== Inference (full denoising loop) ========================

    @torch.no_grad()
    def inference(
        self,
        # Raw inputs
        prompt: Optional[Union[str, List[str]]] = None,
        # Pre-encoded inputs
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        # Reference conditioning
        reference_latents: Optional[torch.Tensor] = None,
        # Generation parameters
        height: int = 256,
        width: int = 416,
        num_frames: int = 33,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # RL-specific
        compute_log_prob: bool = True,
        trajectory_indices: Any = "all",
        extra_call_back_kwargs: List[str] = [],
        # Passthrough
        video_id: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> List["LTXUnionSample"]:
        """Full denoising inference loop with IC-LoRA conditioning."""
        from diffusers.utils.torch_utils import randn_tensor
        from ...utils.trajectory_collector import (
            create_trajectory_collector,
            create_callback_collector,
            TrajectoryIndicesType,
        )

        device = self.device

        # 1. Encode prompt if needed
        if prompt_embeds is None:
            assert prompt is not None, "Either prompt or prompt_embeds required"
            encoded = self.encode_prompt(prompt=prompt)
            prompt_embeds = encoded["prompt_embeds"]
            prompt_ids = encoded["prompt_ids"]
            prompt_attention_mask = encoded["prompt_attention_mask"]
            kwargs["audio_prompt_embeds"] = encoded.get("audio_prompt_embeds")

        batch_size = prompt_embeds.shape[0]
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is None:
            prompt = [""] * batch_size
        if video_id is None:
            video_id = [None] * batch_size
        elif isinstance(video_id, str):
            video_id = [video_id]

        # 2. Prepare noise latents [B, 128, F, H, W]
        F_lat = (num_frames - 1) // 8 + 1
        H_lat = height // 32
        W_lat = width // 32
        latent_shape = (batch_size, 128, F_lat, H_lat, W_lat)
        latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=torch.bfloat16)

        # 3. Patchify reference latents
        patchifier = self.get_component_unwrapped("patchifier")
        assert reference_latents is not None, "reference_latents required for IC-LoRA"
        ref_latents_3d = patchifier.patchify(reference_latents)
        seq_ref = ref_latents_3d.shape[1]

        # 4. Set sigma schedule
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps

        # 5. Trajectory collectors
        latent_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)
        latent_collector.collect(latents, step_idx=0)

        log_prob_collector = None
        if compute_log_prob:
            log_prob_collector = create_trajectory_collector(trajectory_indices, num_inference_steps)

        callback_collector = create_callback_collector(trajectory_indices, num_inference_steps)

        # 6. Denoising loop (C1 fix: include final step to terminal sigma=0)
        for i in range(len(timesteps)):
            t = timesteps[i]
            t_next = (
                timesteps[i + 1]
                if i + 1 < len(timesteps)
                else torch.tensor(0.0, device=device, dtype=timesteps.dtype)
            )
            noise_level_val = self.pipeline.scheduler.get_noise_level_for_timestep(t)
            current_compute_log_prob = compute_log_prob and noise_level_val > 0

            output = self.forward(
                t=t,
                t_next=t_next,
                latents=latents,
                reference_latents=ref_latents_3d,
                ref_seq_len=seq_ref,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                noise_level=noise_level_val,
                compute_log_prob=current_compute_log_prob,
                return_kwargs=["next_latents", "log_prob", "noise_pred"],
                **kwargs,
            )

            latents = output.next_latents
            latent_collector.collect(latents, i + 1)
            if current_compute_log_prob and log_prob_collector is not None:
                log_prob_collector.collect(output.log_prob, i)
            callback_collector.collect_step(
                i, output, extra_call_back_kwargs,
                capturable={"noise_level": noise_level_val},
            )

        # 7. Decode to pixel video
        videos = self.decode_latents(latents, output_type="pt")

        # 8. Package into samples
        all_latents = latent_collector.get_result()
        latent_index_map = latent_collector.get_index_map()
        all_log_probs = log_prob_collector.get_result() if log_prob_collector else None
        log_prob_index_map = log_prob_collector.get_index_map() if log_prob_collector else None
        extra_cb_res = callback_collector.get_result()
        callback_index_map = callback_collector.get_index_map()

        samples = []
        for b in range(batch_size):
            sample = LTXUnionSample(
                timesteps=timesteps,
                all_latents=torch.stack([lat[b] for lat in all_latents], dim=0) if all_latents else None,
                log_probs=torch.stack([lp[b] for lp in all_log_probs], dim=0) if all_log_probs else None,
                latent_index_map=latent_index_map,
                log_prob_index_map=log_prob_index_map,
                video=videos[b] if videos is not None else None,
                height=height,
                width=width,
                prompt=prompt[b] if prompt else "",
                prompt_ids=prompt_ids[b] if prompt_ids is not None else None,
                prompt_embeds=prompt_embeds[b],
                ref_seq_len=seq_ref,
                video_id=video_id[b],
                reference_latents=ref_latents_3d[b],
                extra_kwargs={
                    **{k: v[b] for k, v in extra_cb_res.items()},
                    "callback_index_map": callback_index_map,
                },
            )
            samples.append(sample)

        return samples
