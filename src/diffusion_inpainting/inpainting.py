"""Known-pixel constrained DDPM inpainting (mask=1 means missing)."""
import torch


def square_mask(size: int = 16, width: int = 10, top: int = 3, left: int = 3, device='cpu'):
    if not (0 <= top <= size - width and 0 <= left <= size - width):
        raise ValueError('Mask must fit inside the image')
    mask = torch.zeros(1, 1, size, size, device=device)
    mask[:, :, top:top + width, left:left + width] = 1
    return mask


@torch.no_grad()
def inpaint_ddpm(model, diffusion, images: torch.Tensor, masks: torch.Tensor, labels: torch.Tensor):
    model.eval()
    masks = masks.to(images.device).expand_as(images)
    x = torch.randn_like(images)
    known_noise = torch.randn_like(images)
    for step in reversed(range(diffusion.steps)):
        abar = diffusion.alpha_bars[step]
        known_xt = abar.sqrt() * images + (1 - abar).sqrt() * known_noise
        x = masks * x + (1 - masks) * known_xt
        x = diffusion.reverse_step(model, x, step, labels)
    return masks * x.clamp(-1, 1) + (1 - masks) * images
