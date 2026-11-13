"""Diffusion schedule, forward noising, and training objective."""
import math
import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, steps: int = 200, schedule: str = 'cosine', device: str = 'cpu'):
        self.steps = steps
        if schedule == 'cosine':
            t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
            abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
            abar = abar / abar[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(1e-4, 0.999).float()
        elif schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, steps)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')
        self.betas = betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = self.alphas.cumprod(0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        noise = torch.randn_like(x0) if noise is None else noise
        abar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * noise, noise

    def loss(self, model, images, labels):
        t = torch.randint(0, self.steps, (images.shape[0],), device=images.device)
        noisy, noise = self.q_sample(images, t)
        return F.mse_loss(model(noisy, t, labels), noise)

    @torch.no_grad()
    def reverse_step(self, model, x, t_index: int, labels):
        t = torch.full((x.shape[0],), t_index, device=x.device, dtype=torch.long)
        abar = self.alpha_bars[t_index]
        mean = (x - self.betas[t_index] / (1 - abar).sqrt() * model(x, t, labels)) / self.alphas[t_index].sqrt()
        if t_index == 0:
            return mean
        prev_abar = self.alpha_bars[t_index - 1]
        variance = self.betas[t_index] * (1 - prev_abar) / (1 - abar)
        return mean + variance.sqrt() * torch.randn_like(x)
