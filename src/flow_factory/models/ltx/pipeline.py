"""
LTX Union Pseudo-Pipeline — flat component container for LTX-2 models.

Not a diffusers DiffusionPipeline subclass. LTX-2 uses ltx_core components
(custom transformer, patchifier, Gemma text encoder) which are incompatible
with the diffusers pipeline interface.

BaseAdapter accesses components via getattr(self.pipeline, name). This class
stores all nn.Modules as direct attributes to satisfy that interface.

See: Flow-Factory guidance/new_model.md § "Pseudo-Pipeline for Non-Diffusers Models"
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn
from safetensors import safe_open

from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)


class LTXUnionPseudoPipeline:
    """Flat component container for LTX-2.3 Union-Control model.

    Components:
        transformer: LTXModel (22B diffusion transformer)
        vae_encoder: VideoEncoder (encodes pixel video → latent)
        vae_decoder: VideoDecoder (decodes latent → pixel video)
        text_encoder: GemmaTextEncoder (Gemma LLM + feature extractor)
        embeddings_processor: EmbeddingsProcessor (connectors, block 3)
        patchifier: VideoLatentPatchifier (5D ↔ 3D token conversion)
        scheduler: None — set by adapter's load_scheduler() override
        config: dict with model metadata
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae_encoder: nn.Module,
        vae_decoder: nn.Module,
        text_encoder: nn.Module,
        embeddings_processor: nn.Module,
        patchifier: Any,
        config: dict,
    ):
        self.transformer = transformer
        self.vae_encoder = vae_encoder
        self.vae_decoder = vae_decoder
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.patchifier = patchifier
        self.scheduler = None  # Set by adapter via load_scheduler() override
        self.config = config

    @property
    def vae(self) -> nn.Module:
        """Proxy for BaseAdapter._mix_precision() compatibility.

        BaseAdapter accesses self.pipeline.vae for dtype casting.
        We proxy to vae_encoder (the heavier component).
        decode_latents() accesses vae_decoder directly.
        """
        return self.vae_encoder

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        union_lora_path: Optional[str] = None,
        gemma_path: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        **kwargs: Any,
    ) -> "LTXUnionPseudoPipeline":
        """Load all LTX-2 components and optionally merge Union-Control LoRA."""
        from ltx_trainer.model_loader import (
            load_transformer,
            load_video_vae_encoder,
            load_video_vae_decoder,
            load_text_encoder,
            load_embeddings_processor,
        )
        from ltx_core.components.patchifiers import VideoLatentPatchifier

        transformer = load_transformer(model_path)
        reference_downscale_factor = 1
        if union_lora_path:
            reference_downscale_factor = _merge_lora_into_model(
                transformer, union_lora_path
            )
        vae_encoder = load_video_vae_encoder(model_path)
        vae_decoder = load_video_vae_decoder(model_path)
        text_encoder = load_text_encoder(gemma_path)
        embeddings_processor = load_embeddings_processor(model_path)
        patchifier = VideoLatentPatchifier(patch_size=1)

        config = {
            "model_type": "ltx-union",
            "model_path": model_path,
            "reference_downscale_factor": reference_downscale_factor,
        }

        return cls(
            transformer=transformer,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            embeddings_processor=embeddings_processor,
            patchifier=patchifier,
            config=config,
        )


def _merge_lora_into_model(model: nn.Module, lora_path: str) -> int:
    """Merge LoRA weights from safetensors into model parameters in-place."""
    reference_downscale_factor = 1
    state_dict = {}
    metadata = {}

    with safe_open(lora_path, framework="pt") as f:
        metadata = f.metadata() or {}
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    if "reference_downscale_factor" in metadata:
        reference_downscale_factor = int(metadata["reference_downscale_factor"])

    lora_pairs = {}
    for key, tensor in state_dict.items():
        if ".lora_A." in key:
            base_key = key.replace(".lora_A.weight", "").replace(".lora_A.", "")
            lora_pairs.setdefault(base_key, {})["A"] = tensor
        elif ".lora_B." in key:
            base_key = key.replace(".lora_B.weight", "").replace(".lora_B.", "")
            lora_pairs.setdefault(base_key, {})["B"] = tensor

    default_alpha = float(metadata.get("lora_alpha", "1.0"))
    merged_count = 0
    model_state = dict(model.named_parameters())

    for base_key, pair in lora_pairs.items():
        if "A" not in pair or "B" not in pair:
            logger.warning(f"Incomplete LoRA pair for {base_key}, skipping")
            continue
        param_key = base_key + ".weight"
        if param_key not in model_state:
            if base_key not in model_state:
                logger.warning(f"No matching param for LoRA key {base_key}, skipping")
                continue
            param_key = base_key
        param = model_state[param_key]
        lora_a = pair["A"].to(param.device, param.dtype)
        lora_b = pair["B"].to(param.device, param.dtype)
        rank = lora_a.shape[0]
        alpha = default_alpha
        scale = alpha / rank
        with torch.no_grad():
            if lora_a.dim() == 2 and lora_b.dim() == 2:
                param.add_(scale * (lora_b @ lora_a))
            else:
                logger.warning(
                    f"Non-2D LoRA tensors for {base_key} "
                    f"(A: {lora_a.shape}, B: {lora_b.shape}), skipping"
                )
                continue
        merged_count += 1

    logger.info(
        f"Merged {merged_count} LoRA pairs from {lora_path} "
        f"(alpha={default_alpha}, reference_downscale_factor={reference_downscale_factor})"
    )
    return reference_downscale_factor
