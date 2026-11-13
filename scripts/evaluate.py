"""Generate class-conditional samples and DDPM/SDS comparison figures."""
import argparse
from pathlib import Path
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from diffusion_inpainting import ConditionalUNet, Diffusion, sample_ddpm, square_mask, inpaint_ddpm, inpaint_sds


def plot_grid(images, columns, path, headings=None):
    rows = len(images) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.5, rows * 1.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.imshow(((images[i, 0].detach().cpu().numpy() + 1) / 2).clip(0, 1), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if headings is not None and i < columns:
            ax.set_title(headings[i], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/mnist_ddpm.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('figures/generated'))
    p.add_argument('--sds-steps', type=int, default=1000)
    p.add_argument('--examples-per-class', type=int, default=2)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    torch.manual_seed(int(checkpoint.get('seed', 42)))
    model = ConditionalUNet().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    diffusion = Diffusion(steps=checkpoint['steps'], device=device)
    diffusion.betas = checkpoint['betas'].to(device)
    diffusion.alphas = 1 - diffusion.betas
    diffusion.alpha_bars = diffusion.alphas.cumprod(0)
    labels = torch.arange(40, device=device) % 10
    samples = sample_ddpm(model, diffusion, labels)
    plot_grid(samples, 10, args.output_dir / 'conditional_samples.png', [f'Digit {i}' for i in range(10)])
    valset = datasets.MNIST('data', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    indices = []
    for digit in range(10):
        indices.extend((valset.targets == digit).nonzero().flatten()[:args.examples_per_class].tolist())
    images = torch.stack([valset[i][0] for i in indices]).to(device)
    classes = torch.tensor([valset[i][1] for i in indices], device=device)
    masks = square_mask(device=device).expand_as(images)
    masked = images * (1 - masks) - masks
    ddpm = inpaint_ddpm(model, diffusion, images, masks, classes)
    # SDS is run per image to match the original experiment's optimization setup.
    sds = torch.cat([inpaint_sds(model, diffusion, images[i:i + 1], masks[i:i + 1],
                                  classes[i:i + 1], steps=args.sds_steps)
                     for i in range(len(images))], dim=0)
    panels = torch.stack([images, masked, ddpm, sds], dim=1).reshape(-1, 1, 16, 16)
    plot_grid(panels, 4, args.output_dir / 'ddpm_vs_sds.png', ['Original', 'Masked', 'DDPM', 'SDS'])
    known = 1 - masks
    print('Known-pixel maximum error (DDPM):', ((ddpm - images) * known).abs().max().item())
    print('Known-pixel maximum error (SDS): ', ((sds - images) * known).abs().max().item())
    print('Figures saved to', args.output_dir)


if __name__ == '__main__':
    main()
