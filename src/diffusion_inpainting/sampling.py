"""Ancestral DDPM sampling."""
import torch


@torch.no_grad()
def sample_ddpm(model, diffusion, labels: torch.Tensor, image_size: int = 16):
    model.eval()
    x = torch.randn(len(labels), 1, image_size, image_size, device=labels.device)
    for step in reversed(range(diffusion.steps)):
        x = diffusion.reverse_step(model, x, step, labels)
    return x.clamp(-1, 1)
