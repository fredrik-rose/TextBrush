"""
Diffusion algorithm.
"""

import typing

import torch

from torch import nn

DDIM_STEP_SIZE = 10


class Diffuser(nn.Module):
    """
    Diffusion process implementing DDPM (Denoising Diffusion Probabilistic Models).
    """

    def __init__(
        self,
        betas: list[float],
    ):
        super().__init__()

        self.time_steps = len(betas)

        betas_tensor = torch.tensor(betas, dtype=torch.float32)
        self.register_buffer("_betas", betas_tensor)
        self.register_buffer("_a", 1 - betas_tensor)
        self.register_buffer("_a_bar", torch.cumprod(1 - betas_tensor, dim=0))
        self.register_buffer("_s", betas_tensor**0.5)

    def forward(self, *args, **kwargs):  # pylint: disable=missing-function-docstring
        raise NotImplementedError("Diffusion is not a neural network, use forward_diffusion() or reverse_diffusion().")

    def forward_diffusion(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create training samples.
        """
        t = torch.randint(low=0, high=self.time_steps, size=(1,), device=x.device)
        e = torch.normal(mean=0, std=1, size=x.shape, device=x.device)
        x = (self._a_bar[t] ** 0.5) * x + ((1 - self._a_bar[t]) ** 0.5) * e
        return x, e, t

    def reverse_diffusion(
        self,
        size: tuple[int],
        condition: int,
        noise_predictor: nn.Module,
        ddim: bool = True,
    ) -> typing.Generator[torch.Tensor, None, None]:
        """
        Sample from the diffusion process.
        """
        device = self._betas.device
        x = torch.normal(mean=0, std=1, size=size, device=device)
        zero = torch.zeros_like(x, device=device)
        yield x
        time_steps = (
            torch.linspace(
                0 if ddim else -1,
                self.time_steps - 1,
                steps=self.time_steps // DDIM_STEP_SIZE if ddim else self.time_steps,
            )
            .long()
            .flip(0)
        )
        for t, t_next in zip(time_steps[:-1], time_steps[1:]):
            e = noise_predictor(
                x,
                torch.tensor([t], dtype=torch.long, device=device).expand(x.size(0), -1),
                torch.tensor([condition], dtype=torch.long, device=device).expand(x.size(0), -1),
            )
            if ddim:  # DDIM
                a_t_next = self._a_bar[t_next]
                x_0 = (self._a_bar[t] ** -0.5) * (x - ((1 - self._a_bar[t]) ** 0.5) * e)
                d_x_t = ((1 - a_t_next) ** 0.5) * e
                x = a_t_next**0.5 * x_0 + d_x_t
            else:  # DDPM
                z = torch.normal(mean=0, std=1, size=size, device=device) if t > 0 else zero
                x = (self._a[t] ** -0.5) * (x - (self._betas[t] / (1 - self._a_bar[t]) ** 0.5) * e) + self._s[t] * z
            yield x


def get_linear_noise_schedule(
    b_1: float = 10e-4,
    b_t: float = 0.02,
    time_steps: int = 1000,
) -> list[float]:
    """
    Create a linear noise variance schedule to be used in the diffusion process.
    """
    step = (b_t - b_1) / (time_steps - 1)
    betas = [b_1 + (step * i) for i in range(time_steps)]
    return betas
