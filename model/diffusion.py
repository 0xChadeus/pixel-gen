"""EDM (Elucidating Diffusion Models) framework.

Implements Karras et al. 2022 preconditioning, loss weighting, noise
sampling, and Heun ODE sampler. This wraps the raw UNet denoiser.

Reference: https://arxiv.org/abs/2206.00364
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EDMPrecond(nn.Module):
    """EDM preconditioning wrapper around the raw denoiser network.

    Applies analytically derived input/output scaling based on noise level sigma.
    The network predicts D(x; sigma) and this wrapper computes:
        F(x; sigma) = c_skip(sigma) * x + c_out(sigma) * D(c_in(sigma) * x; c_noise(sigma))
    """

    def __init__(self, model: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        cross_tokens: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, C, H, W) noisy sample
            sigma: (B,) noise level
            cond: (B, cond_dim) FiLM conditioning
            cross_tokens: (B, T, D) text tokens
            x_self_cond: (B, C, H, W) optional self-conditioning input

        Returns:
            (B, C, H, W) denoised estimate
        """
        sigma = sigma.reshape(-1, 1, 1, 1)
        sd = self.sigma_data

        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in = 1.0 / (sigma ** 2 + sd ** 2).sqrt()
        c_noise = sigma.squeeze().log() / 4.0  # used as conditioning

        # The raw denoiser takes the scaled input
        D = self.model(
            c_in * x_noisy,
            cond=cond,
            cross_tokens=cross_tokens,
            x_self_cond=x_self_cond,
        )

        return c_skip * x_noisy + c_out * D


class EDMLoss(nn.Module):
    """EDM training loss with optimal weighting and optional LPIPS.

    Samples noise levels from log-normal distribution and applies
    the EDM loss weighting lambda(sigma). Optionally adds LPIPS
    perceptual loss as an auxiliary term for improved pixel quality.
    """

    def __init__(self, sigma_data: float = 0.5, P_mean: float = -1.2,
                 P_std: float = 1.2, lpips_weight: float = 0.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.lpips_weight = lpips_weight

        self._lpips_fn = None
        if lpips_weight > 0:
            import lpips
            self._lpips_fn = lpips.LPIPS(net="alex", verbose=False)
            self._lpips_fn.requires_grad_(False)

    def forward(
        self,
        precond_model: EDMPrecond,
        x_clean: torch.Tensor,
        cond: torch.Tensor,
        cross_tokens: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
        resolution: int = 64,
    ) -> torch.Tensor:
        """Compute weighted EDM loss.

        Args:
            precond_model: EDMPrecond wrapper
            x_clean: (B, C, H, W) clean training sample
            cond: (B, cond_dim) conditioning
            cross_tokens: (B, T, D) text tokens
            x_self_cond: optional self-conditioning
            resolution: image resolution for noise schedule adjustment

        Returns:
            Scalar loss
        """
        B = x_clean.shape[0]

        # Resolution-aware noise schedule: shift P_mean based on resolution
        # Higher resolution = more pixel redundancy = shift toward noisier
        P_mean_adj = self.P_mean + 0.5 * math.log(resolution / 64)

        # Sample noise levels from log-normal
        log_sigma = torch.randn(B, device=x_clean.device) * self.P_std + P_mean_adj
        sigma = log_sigma.exp()

        # Loss weighting
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Add noise
        noise = torch.randn_like(x_clean)
        x_noisy = x_clean + sigma.reshape(-1, 1, 1, 1) * noise

        # Get denoised estimate
        denoised = precond_model(x_noisy, sigma, cond, cross_tokens, x_self_cond)

        # Weighted MSE loss
        mse_loss = weight.reshape(-1, 1, 1, 1) * (denoised - x_clean) ** 2
        total_loss = mse_loss.mean()

        # LPIPS perceptual loss (only for low-noise samples where it's meaningful)
        if self._lpips_fn is not None and self.lpips_weight > 0:
            low_noise = sigma < 1.0
            if low_noise.any():
                self._lpips_fn = self._lpips_fn.to(x_clean.device)
                # Convert OKLab (4ch) to RGB-like (3ch) for LPIPS
                # Use first 3 channels (OKLab) scaled to roughly [-1, 1]
                denoised_rgb = denoised[low_noise, :3]
                clean_rgb = x_clean[low_noise, :3]
                lpips_val = self._lpips_fn(denoised_rgb, clean_rgb).mean()
                total_loss = total_loss + self.lpips_weight * lpips_val

        return total_loss


class HeunSampler:
    """EDM 2nd-order Heun ODE sampler.

    Produces higher quality samples than Euler with the same number of
    function evaluations (each Heun step = 2 NFEs but much better accuracy).
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        S_churn: float = 0.0,
        S_noise: float = 1.0,
        S_tmin: float = 0.0,
        S_tmax: float = float("inf"),
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_noise = S_noise
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax

    def get_sigmas(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Generate the noise level schedule."""
        rho = self.rho
        min_inv = self.sigma_min ** (1 / rho)
        max_inv = self.sigma_max ** (1 / rho)
        steps = torch.arange(num_steps + 1, device=device, dtype=torch.float32)
        sigmas = (max_inv + steps / num_steps * (min_inv - max_inv)) ** rho
        sigmas[-1] = 0  # final step is clean
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        precond_model: EDMPrecond,
        shape: tuple[int, ...],
        cond: torch.Tensor,
        cross_tokens: torch.Tensor,
        num_steps: int = 35,
        device: torch.device | None = None,
        progress_callback=None,
        self_condition: bool = False,
    ) -> torch.Tensor:
        """Sample from the model using Heun's method.

        Args:
            precond_model: EDMPrecond wrapper
            shape: (B, C, H, W) output shape
            cond: (B, cond_dim) conditioning
            cross_tokens: (B, T, D) text tokens
            num_steps: Number of sampling steps
            device: Target device
            progress_callback: Called with (step, total_steps) at each step
            self_condition: Whether to use self-conditioning

        Returns:
            (B, C, H, W) generated samples
        """
        if device is None:
            device = cond.device

        sigmas = self.get_sigmas(num_steps, device)

        # Start from pure noise at sigma_max
        x = torch.randn(shape, device=device) * sigmas[0]
        x_self_cond = None

        for i in range(num_steps):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Optional stochastic noise injection
            gamma = 0.0
            if self.S_tmin <= sigma_cur <= self.S_tmax:
                gamma = min(self.S_churn / num_steps, math.sqrt(2) - 1)

            sigma_hat = sigma_cur * (1 + gamma)
            if gamma > 0:
                noise = torch.randn_like(x) * self.S_noise
                x = x + (sigma_hat ** 2 - sigma_cur ** 2).sqrt() * noise

            # Self-conditioning: use previous denoised estimate
            sc = x_self_cond if self_condition else None

            # Denoised estimate at sigma_hat
            sigma_batch = torch.full((shape[0],), sigma_hat, device=device)
            denoised = precond_model(x, sigma_batch, cond, cross_tokens, sc)

            if self_condition:
                x_self_cond = denoised.detach()

            # Euler step
            d = (x - denoised) / sigma_hat
            x_next = x + d * (sigma_next - sigma_hat)

            # Heun correction (2nd order) if not the last step
            if sigma_next > 0:
                sigma_batch_next = torch.full((shape[0],), sigma_next, device=device)
                denoised_2 = precond_model(x_next, sigma_batch_next, cond, cross_tokens,
                                           x_self_cond if self_condition else None)
                d_2 = (x_next - denoised_2) / sigma_next
                x_next = x + (d + d_2) / 2 * (sigma_next - sigma_hat)

            x = x_next

            if progress_callback is not None:
                progress_callback(i + 1, num_steps)

        return x


class CFGWrapper:
    """Classifier-free guidance wrapper for sampling."""

    def __init__(self, precond_model: EDMPrecond, guidance_scale: float = 5.0):
        self.model = precond_model
        self.scale = guidance_scale

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
        cross_tokens: torch.Tensor,
        cross_tokens_uncond: torch.Tensor,
        x_self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Conditional prediction
        pred_cond = self.model(x, sigma, cond, cross_tokens, x_self_cond)
        # Unconditional prediction
        pred_uncond = self.model(x, sigma, uncond, cross_tokens_uncond, x_self_cond)
        # Guided output
        return pred_uncond + self.scale * (pred_cond - pred_uncond)
