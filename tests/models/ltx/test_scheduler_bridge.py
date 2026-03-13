import pytest
import torch
import math

class TestLTXSDEScheduler:
    def test_construction(self):
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=10)
        assert scheduler is not None

    def test_set_timesteps_produces_decreasing_sigmas(self):
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=10)
        scheduler.set_timesteps(10, device="cpu")
        for i in range(len(scheduler.sigmas) - 1):
            assert scheduler.sigmas[i] >= scheduler.sigmas[i + 1]

    def test_ode_euler_step(self):
        """ODE step: next = latent + (sigma_next - sigma) * v_pred"""
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=4)
        scheduler.set_timesteps(4, device="cpu")
        scheduler.dynamics_type = "ODE"

        latents = torch.randn(2, 128, 1, 4, 4)
        v_pred = torch.randn_like(latents)
        sigma = torch.tensor(0.8)
        sigma_next = torch.tensor(0.4)

        output = scheduler.step(
            noise_pred=v_pred,
            timestep=sigma * 1000,
            latents=latents,
            timestep_next=sigma_next * 1000,
            noise_level=0.0,
            compute_log_prob=False,
            return_dict=True,
        )

        dt = sigma_next - sigma  # negative
        expected = latents + v_pred * dt
        torch.testing.assert_close(
            output.next_latents, expected.float(), atol=1e-6, rtol=1e-6
        )

    def test_sde_step_adds_noise(self):
        """SDE step should differ from ODE step (noise injected)."""
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=4)
        scheduler.set_timesteps(4, device="cpu")
        scheduler.dynamics_type = "Flow-SDE"

        latents = torch.randn(2, 128, 1, 4, 4)
        v_pred = torch.randn_like(latents)
        sigma = torch.tensor(0.8)
        sigma_next = torch.tensor(0.4)
        gen = torch.Generator().manual_seed(42)

        output = scheduler.step(
            noise_pred=v_pred,
            timestep=sigma * 1000,
            latents=latents,
            timestep_next=sigma_next * 1000,
            noise_level=0.5,
            compute_log_prob=True,
            return_dict=True,
            generator=gen,
        )

        ode_result = latents.float() + v_pred.float() * (sigma_next - sigma)
        assert not torch.equal(output.next_latents, ode_result)
        assert output.log_prob is not None
        assert output.log_prob.shape == (2,)

    def test_get_noise_level_for_timestep(self):
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=10)
        scheduler.set_timesteps(10, device="cpu")
        assert scheduler.get_noise_level_for_sigma(torch.tensor(0.0)) == 0.0

    def test_distilled_sigma_schedule(self):
        """When using distilled sigmas, should use DISTILLED_SIGMA_VALUES."""
        from flow_factory.models.ltx.scheduler_bridge import LTXSDEScheduler
        scheduler = LTXSDEScheduler(num_inference_steps=4, use_distilled=True)
        scheduler.set_timesteps(4, device="cpu")
        assert len(scheduler.sigmas) > 0
