"""Train the class-conditional DDPM on MNIST."""
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion_inpainting import ConditionalUNet, Diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=22)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/mnist_ddpm.pt'))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = datasets.MNIST('data', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((16, 16)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = ConditionalUNet().to(device)
    diffusion = Diffusion(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.loss(model, images, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f'Epoch {epoch + 1}/{args.epochs} | noise MSE: {running / len(loader):.5f}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'steps': diffusion.steps,
                'betas': diffusion.betas.cpu(), 'seed': args.seed}, args.output)
    print(f'Saved checkpoint: {args.output}')


if __name__ == '__main__':
    main()
