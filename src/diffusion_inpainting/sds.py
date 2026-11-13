"""SDS-style optimization of image pixels using a frozen denoiser."""
import torch


def sds_gradient(model, diffusion, x0, labels, min_fraction=0.05, max_fraction=0.95):
    """Detached score-distillation direction; no gradient through the denoiser."""
    batch = x0.shape[0]
    t = torch.randint(int(diffusion.steps * min_fraction), int(diffusion.steps * max_fraction),
                      (batch,), device=x0.device)
    xt, noise = diffusion.q_sample(x0, t)
    with torch.no_grad():
        prediction = model(xt, t, labels)
    weight = 1 - diffusion.alpha_bars[t].view(-1, 1, 1, 1)
    return (weight * (prediction - noise)).detach()


def inpaint_sds(model, diffusion, images, masks, labels, steps=1000, lr=1e-2):
    """Optimize missing pixels only; known pixels are copied from images exactly."""
    model.eval()
    masks = masks.to(images.device).expand_as(images)
    # Use an independent tensor; do not modify the observed images or model parameters.
    pixels = torch.zeros_like(images, requires_grad=True)
    optimizer = torch.optim.Adam([pixels], lr=lr)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        composed = masks * pixels + (1 - masks) * images
        grad = sds_gradient(model, diffusion, composed, labels)
        surrogate = (grad * composed).sum() / len(images)
        surrogate.backward()
        optimizer.step()
        with torch.no_grad():
            pixels.clamp_(-1, 1)
    return (masks * pixels.detach() + (1 - masks) * images).detach()
