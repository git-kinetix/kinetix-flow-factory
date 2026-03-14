# Copyright 2026 Jayce-Ping
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/flow_factory/models/ltx/scheduler_bridge.py
"""
LTXSDEScheduler: bridges LTX-2's sigma-based scheduling to Flow-Factory's
SDESchedulerMixin interface.

Key design decision:
    LTX-2 uses sigma ∈ [0, 1] directly.
    Flow-Factory uses timesteps = sigma * 1000.
    This scheduler accepts both, converting transparently.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union
import math

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from ...scheduler.abc import SDESchedulerMixin, SDESchedulerOutput
from ...utils.base import to_broadcast_tensor
from ...utils.logger_utils import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Distilled sigma schedule (used when use_distilled=True)
# ---------------------------------------------------------------------------
# Exact distilled sigma schedule from ltx_pipelines/utils/constants.py.
# These 9 values (8 steps) match the original ICLoraPipeline Stage 1.
DISTILLED_SIGMA_VALUES: List[float] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------
@dataclass
class LTXSDESchedulerOutput(SDESchedulerOutput):
    """Output class for a single LTX SDE step."""
    pass


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
class LTXSDEScheduler(SDESchedulerMixin):
    """
    Scheduler that bridges LTX-2's direct-sigma flow-matching to
    Flow-Factory's SDESchedulerMixin interface.

    Supports ODE, Flow-SDE, Dance-SDE and CPS dynamics.

    Timestep convention (FF compatibility):
        timestep = sigma * 1000
        sigma    = timestep / 1000
    """

    def __init__(
        self,
        num_inference_steps: int = 20,
        noise_level: float = 0.7,
        sde_steps: Optional[Union[int, List[int], torch.Tensor]] = None,
        num_sde_steps: Optional[int] = None,
        seed: int = 42,
        dynamics_type: Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"] = "Flow-SDE",
        use_distilled: bool = False,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
    ):
        self.num_inference_steps = num_inference_steps
        self.noise_level = noise_level
        assert self.noise_level >= 0, "Noise level must be non-negative."

        self._sde_steps = (
            torch.tensor(sde_steps, dtype=torch.int64)
            if sde_steps is not None
            else None
        )
        self._num_sde_steps = num_sde_steps
        self.seed = seed
        self.dynamics_type = dynamics_type
        self.use_distilled = use_distilled
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self._is_eval = False

        # Will be populated by set_timesteps()
        self.sigmas: torch.Tensor = torch.tensor([])
        self.timesteps: torch.Tensor = torch.tensor([])

    # ------------------------------------------------------------------
    # Public API: timestep setup
    # ------------------------------------------------------------------
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[Union[str, torch.device]] = None,
        sigmas: Optional[Union[List[float], np.ndarray]] = None,
    ) -> None:
        """
        Set the sigma / timestep schedule.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Target device for tensors.
            sigmas: Optional custom sigma sequence (overrides defaults).
        """
        self.num_inference_steps = num_inference_steps

        if sigmas is not None:
            sigma_values = list(sigmas)
        elif self.use_distilled:
            # Use the exact distilled schedule. The schedule has len(src)-1 steps.
            src = DISTILLED_SIGMA_VALUES
            num_distilled_steps = len(src) - 1  # 8 steps for 9 sigma values
            if num_inference_steps >= num_distilled_steps:
                sigma_values = src
            else:
                # Sub-sample: evenly spaced indices including first and last
                indices = np.round(
                    np.linspace(0, len(src) - 1, num_inference_steps + 1)
                ).astype(int)
                sigma_values = [src[i] for i in indices]
        else:
            # Use LTX2Scheduler (shifted + stretched) to match original pipeline
            try:
                from ltx_core.components.schedulers import LTX2Scheduler
                ltx_sched = LTX2Scheduler()
                sigma_tensor = ltx_sched.execute(steps=num_inference_steps)
                sigma_values = sigma_tensor.tolist()
            except ImportError:
                # Fallback to linear schedule if ltx_core not available
                sigma_values = list(
                    np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1)
                )

        # Ensure monotonically non-increasing
        self.sigmas = torch.tensor(sigma_values, dtype=torch.float32, device=device)

        # Timesteps = sigma * 1000  (FF convention)
        # We expose the "active" timesteps (excluding the trailing 0)
        self.timesteps = (self.sigmas[:-1] * 1000).to(device=device)

    # ------------------------------------------------------------------
    # Mode management (SDESchedulerMixin interface)
    # ------------------------------------------------------------------
    @property
    def is_eval(self) -> bool:
        return self._is_eval

    def eval(self) -> None:
        """Switch to deterministic ODE sampling."""
        self._is_eval = True

    def train(self, mode: bool = True) -> None:
        """Switch to stochastic SDE sampling."""
        self._is_eval = not mode

    def rollout(self, mode: bool = True) -> None:
        """Alias for train()."""
        self.train(mode=mode)

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    # ------------------------------------------------------------------
    # Step selection (SDESchedulerMixin interface)
    # ------------------------------------------------------------------
    @property
    def sde_steps(self) -> torch.Tensor:
        """Step indices eligible for SDE noise injection."""
        if self._sde_steps is not None:
            if not isinstance(self._sde_steps, torch.Tensor):
                self._sde_steps = torch.tensor(self._sde_steps, dtype=torch.int64)
            return self._sde_steps
        # Default: all steps except the last
        return torch.arange(0, len(self.timesteps) - 1, dtype=torch.int64)

    @property
    def num_sde_steps(self) -> int:
        if self._num_sde_steps is not None:
            return self._num_sde_steps
        return len(self.sde_steps)

    @property
    def current_sde_steps(self) -> torch.Tensor:
        if self.num_sde_steps >= len(self.sde_steps):
            return self.sde_steps
        generator = torch.Generator().manual_seed(self.seed)
        selected = torch.randperm(len(self.sde_steps), generator=generator)[: self.num_sde_steps]
        return self.sde_steps[selected]

    @property
    def train_timesteps(self) -> torch.Tensor:
        return self.current_sde_steps

    def get_train_timesteps(self) -> torch.Tensor:
        return self.timesteps[self.train_timesteps]

    def get_train_sigmas(self) -> torch.Tensor:
        return self.sigmas[self.train_timesteps]

    # ------------------------------------------------------------------
    # Noise level helpers (SDESchedulerMixin interface)
    # ------------------------------------------------------------------
    def get_noise_levels(self) -> torch.Tensor:
        noise_levels = torch.zeros_like(self.timesteps, dtype=torch.float32)
        noise_levels[self.current_sde_steps] = self.noise_level
        return noise_levels

    def get_noise_level_for_timestep(
        self, timestep: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        if not isinstance(timestep, torch.Tensor) or timestep.ndim == 0:
            t = timestep.item() if isinstance(timestep, torch.Tensor) else timestep
            idx = self._index_for_timestep(t)
            return self.noise_level if idx in self.current_sde_steps else 0.0

        indices = torch.tensor([self._index_for_timestep(t.item()) for t in timestep])
        mask = torch.isin(indices, self.current_sde_steps)
        return torch.where(mask, self.noise_level, 0.0).to(timestep.dtype)

    def get_noise_level_for_sigma(
        self, sigma: Union[float, torch.Tensor]
    ) -> Union[float, torch.Tensor]:
        """Return noise level for the given sigma value(s)."""
        if not isinstance(sigma, torch.Tensor):
            sigma_tensor = torch.tensor(
                [sigma], device=self.sigmas.device, dtype=self.sigmas.dtype
            )
            is_scalar = True
        else:
            sigma_tensor = sigma.to(device=self.sigmas.device, dtype=self.sigmas.dtype)
            is_scalar = sigma_tensor.ndim == 0
            if is_scalar:
                sigma_tensor = sigma_tensor.unsqueeze(0)

        # Check if sigma == 0: it's the terminal sigma (not an SDE step)
        match_mask = (sigma_tensor.unsqueeze(-1) == self.sigmas.unsqueeze(0))

        if not match_mask.any(dim=-1).all():
            # Sigma not found in schedule — return 0.0 (safe fallback)
            result = torch.zeros(sigma_tensor.shape, dtype=sigma_tensor.dtype, device=sigma_tensor.device)
            return result.item() if is_scalar else result

        indices = match_mask.int().argmax(dim=-1)
        sde_mask = torch.isin(indices, self.current_sde_steps.to(indices.device))
        result = torch.where(
            sde_mask,
            torch.tensor(self.noise_level, dtype=sigma_tensor.dtype, device=sigma_tensor.device),
            torch.tensor(0.0, dtype=sigma_tensor.dtype, device=sigma_tensor.device),
        )
        return result.item() if is_scalar else result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _index_for_timestep(self, timestep: float) -> int:
        """Return the step index for a scalar timestep value."""
        candidates = (self.timesteps == timestep).nonzero(as_tuple=False)
        if len(candidates) == 0:
            raise ValueError(f"Timestep {timestep} not found in scheduler timesteps.")
        return int(candidates[0].item())

    # ------------------------------------------------------------------
    # Core step  (SDESchedulerMixin interface)
    # ------------------------------------------------------------------
    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        latents: torch.Tensor,
        next_latents: Optional[torch.Tensor] = None,
        timestep_next: Optional[Union[float, torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        noise_level: Optional[Union[int, float, torch.Tensor]] = None,
        compute_log_prob: bool = True,
        return_dict: bool = True,
        return_kwargs: List[str] = [
            "next_latents",
            "next_latents_mean",
            "std_dev_t",
            "dt",
            "log_prob",
            "noise_pred",
        ],
        dynamics_type: Optional[Literal["Flow-SDE", "Dance-SDE", "CPS", "ODE"]] = None,
        sigma_max: Optional[float] = None,
    ) -> Union[LTXSDESchedulerOutput, Tuple]:
        """
        Perform one denoising step.

        Timesteps follow FF convention: timestep = sigma * 1000.
        When `timestep_next` is provided, sigmas are derived directly:
            sigma      = timestep      / 1000
            sigma_prev = timestep_next / 1000
        """
        # -- Resolve sigma / sigma_prev --
        # Prefer schedule lookup to avoid sigma*1000/1000 float32 roundtrip error.
        t_val = timestep.item() if isinstance(timestep, torch.Tensor) else float(timestep)
        try:
            idx = self._index_for_timestep(t_val)
            sigma = self.sigmas[idx]
            sigma_prev = self.sigmas[idx + 1] if idx + 1 < len(self.sigmas) else torch.tensor(0.0)
        except ValueError:
            # Timestep not in schedule (e.g., arbitrary timestep in training)
            sigma = torch.tensor(t_val / 1000.0)
            if timestep_next is not None:
                t_next_val = timestep_next.item() if isinstance(timestep_next, torch.Tensor) else float(timestep_next)
                sigma_prev = torch.tensor(t_next_val / 1000.0)
            else:
                sigma_prev = torch.tensor(0.0)

        # -- Dynamics type & noise level --
        dynamics_type = dynamics_type or self.dynamics_type
        if self.is_eval or dynamics_type == "ODE":
            noise_level = 0.0
        elif noise_level is None:
            noise_level = self.get_noise_level_for_sigma(sigma)

        # -- Numerical promotion --
        # ODE mode: preserve native dtype to match original EulerDiffusionStep
        # SDE modes: promote to float32 for precise log_prob computation
        if dynamics_type != "ODE":
            noise_pred = noise_pred.float()
            latents = latents.float()
            if next_latents is not None:
                next_latents = next_latents.float()

        noise_level = to_broadcast_tensor(noise_level, latents)
        # For ODE mode: keep sigma in float32 to avoid bf16 rounding
        # (e.g., sigma=0.9999998 rounds to 1.0 in bf16, changing the step result).
        # Use a float32 reference tensor for broadcasting shape.
        if dynamics_type == "ODE":
            ref_for_broadcast = latents.float()
        else:
            ref_for_broadcast = latents
        sigma      = to_broadcast_tensor(sigma,      ref_for_broadcast)
        sigma_prev = to_broadcast_tensor(sigma_prev, ref_for_broadcast)
        dt = sigma_prev - sigma   # negative scalar (flow direction)

        # -- Compute next sample --
        log_prob: Optional[torch.Tensor] = None

        if dynamics_type == "ODE":
            # Replicate original ltx_core roundtrip (3 separate f32→bf16 casts):
            #   to_denoised: x0 = (sample.f32 - v.f32 * sigma.f32).bf16
            #   to_velocity: v' = ((sample.f32 - x0.f32) / sigma.item()).bf16
            #   euler step:  next = (sample.f32 + v'.f32 * dt).bf16
            # NOTE: to_velocity uses sigma.item() (Python float), NOT tensor division.
            # On CUDA, f32/python_float uses a different kernel than f32/f32_tensor,
            # producing slightly different results (~6e-8) that round to different
            # bf16 values. We must use .item() to match the original exactly.
            orig_dtype = latents.dtype
            sigma_item = sigma.float().flatten()[0].item()
            x0 = (latents.float() - noise_pred.float() * sigma.float()).to(orig_dtype)
            velocity = ((latents.float() - x0.float()) / sigma_item).to(orig_dtype)
            next_latents_mean = (latents.float() + velocity.float() * dt.float()).to(orig_dtype)
            std_dev_t = torch.zeros_like(sigma)

            if next_latents is None:
                next_latents = next_latents_mean

            if compute_log_prob:
                logger.warning(
                    "`log_prob` is meaningless when `dynamics_type` is 'ODE', setting to zero."
                )
                log_prob = torch.zeros(
                    (next_latents.shape[0],),
                    dtype=next_latents.dtype,
                    device=next_latents.device,
                )

        elif dynamics_type == "Flow-SDE":
            _sigma_max = sigma_max or self.sigmas[1].item() if len(self.sigmas) > 1 else 1.0
            _sigma_max = to_broadcast_tensor(_sigma_max, latents)
            std_dev_t = (
                torch.sqrt(sigma / (1 - torch.where(sigma == 1.0, _sigma_max, sigma)))
                * noise_level
            )

            next_latents_mean = (
                latents * (1 + std_dev_t**2 / (2 * sigma) * dt)
                + noise_pred * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
            )

            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

            if compute_log_prob:
                std_variance = std_dev_t * torch.sqrt(-1 * dt)
                log_prob = (
                    -((next_latents.detach() - next_latents_mean) ** 2) / (2 * std_variance**2)
                    - torch.log(std_variance)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif dynamics_type == "Dance-SDE":
            pred_original_sample = latents - sigma * noise_pred
            std_dev_t = noise_level * torch.sqrt(-1 * dt)
            log_term = (
                0.5
                * noise_level**2
                * (latents - pred_original_sample * (1 - sigma))
                / sigma**2
            )
            next_latents_mean = latents + (noise_pred + log_term) * dt

            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * variance_noise

            if compute_log_prob:
                log_prob = (
                    (
                        -((next_latents.detach() - next_latents_mean) ** 2)
                        / (2 * (std_dev_t**2))
                    )
                    - torch.log(std_dev_t)
                    - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
                )
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        elif dynamics_type == "CPS":
            std_dev_t = sigma_prev * torch.sin(noise_level * torch.pi / 2)
            x0 = latents - sigma * noise_pred
            x1 = latents + noise_pred * (1 - sigma)
            next_latents_mean = (
                x0 * (1 - sigma_prev)
                + x1 * torch.sqrt(sigma_prev**2 - std_dev_t**2)
            )

            if next_latents is None:
                variance_noise = randn_tensor(
                    noise_pred.shape,
                    generator=generator,
                    device=noise_pred.device,
                    dtype=noise_pred.dtype,
                )
                next_latents = next_latents_mean + std_dev_t * variance_noise

            if compute_log_prob:
                log_prob = -((next_latents.detach() - next_latents_mean) ** 2)
                log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

        else:
            raise ValueError(
                f"Unknown dynamics_type '{dynamics_type}'. "
                "Choose from 'ODE', 'Flow-SDE', 'Dance-SDE', 'CPS'."
            )

        if not compute_log_prob:
            log_prob = None

        if not return_dict:
            return (next_latents, next_latents_mean, noise_pred, log_prob, std_dev_t, dt)

        d: dict = {}
        local_vars = locals()
        for k in return_kwargs:
            if k in local_vars:
                d[k] = local_vars[k]
            else:
                logger.warning(f"Requested return keyword '{k}' is not available in step output.")

        return LTXSDESchedulerOutput.from_dict(d)
