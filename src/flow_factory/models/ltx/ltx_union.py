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

    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError("Implemented in Task 6")

    # ======================== Sampling & Training (stubs) ========================

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 6")

    def inference(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 7")
