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

    # ======================== Encoding (stubs — implemented in Task 5) ========================

    def encode_prompt(self, prompt, **kwargs):
        raise NotImplementedError("Implemented in Task 5")

    def encode_image(self, images, **kwargs):
        return None  # Not used; conditioning via encode_video

    def encode_video(self, videos, **kwargs):
        raise NotImplementedError("Implemented in Task 5")

    def decode_latents(self, latents, **kwargs):
        raise NotImplementedError("Implemented in Task 6")

    # ======================== Sampling & Training (stubs) ========================

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 6")

    def inference(self, *args, **kwargs):
        raise NotImplementedError("Implemented in Task 7")
