import torch
from diffusion_inpainting import ConditionalUNet, Diffusion, square_mask, inpaint_ddpm, inpaint_sds, sample_ddpm


def test_model_shapes():
    model = ConditionalUNet()
    out = model(torch.randn(2, 1, 16, 16), torch.tensor([0, 9]), torch.tensor([3, 5]))
    assert out.shape == (2, 1, 16, 16)


def test_schedule_and_forward_noise():
    diffusion = Diffusion(steps=12)
    assert torch.all(diffusion.alpha_bars[1:] < diffusion.alpha_bars[:-1])
    x = torch.randn(2, 1, 16, 16)
    xt, eps = diffusion.q_sample(x, torch.tensor([0, 11]))
    assert xt.shape == eps.shape == x.shape


def test_sampling_and_preservation():
    torch.manual_seed(0)
    model = ConditionalUNet()
    diffusion = Diffusion(steps=3)
    images = torch.randn(2, 1, 16, 16).clamp(-1, 1)
    labels = torch.tensor([0, 1])
    mask = square_mask()
    assert sample_ddpm(model, diffusion, labels).shape == images.shape
    ddpm = inpaint_ddpm(model, diffusion, images, mask, labels)
    sds = inpaint_sds(model, diffusion, images, mask, labels, steps=2)
    known = 1 - mask
    assert torch.equal(ddpm * known, images * known)
    assert torch.equal(sds * known, images * known)
    assert torch.isfinite(ddpm).all() and torch.isfinite(sds).all()
