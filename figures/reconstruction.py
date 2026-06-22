"""Reconstruction quality visualization.

Encodes a batch of images and decodes them back, then shows
  original | reconstruction
side-by-side for each image.

Usage:
    uv run python reconstruction.py --ckpt runs/cifar10_v1_last.pt
    uv run python reconstruction.py --ckpt runs/cifar10_v1_last.pt img1.png img2.png
    uv run python reconstruction.py --ckpt runs/celeba64_last.pt --config configs/celeba64.yaml --dataset celeba64 --n 16
"""

import argparse
import os

import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from PIL import Image

from src.model import AutoEncoder
from src.utils import load_config


def load_image(path, size, device):
    img = Image.open(path).convert('RGB')
    img = TF.resize(img, [size, size])
    return TF.to_tensor(img).unsqueeze(0).to(device)


def get_dataset_images(dataset_name, data_path, n, size, device):
    """Load n images from the validation/test split of a torchvision dataset."""
    transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor()])

    if dataset_name == 'cifar10':
        ds = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform)
    elif dataset_name == 'mnist':
        ds = torchvision.datasets.MNIST(
            root=data_path, train=False, download=True, transform=T.Compose([
                T.Pad(2), T.ToTensor()]))
    elif dataset_name == 'celeba64':
        ds = torchvision.datasets.CelebA(
            root=data_path, split='valid', download=False, transform=transform)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    loader = torch.utils.data.DataLoader(ds, batch_size=n, shuffle=False)
    imgs, _ = next(iter(loader))
    return imgs.to(device)


def to_01(imgs, decoder_type):
    if decoder_type != 'bernoulli':
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--config',     default='configs/cifar10.yaml')
    parser.add_argument('--dataset',    default='cifar10',
                        choices=['cifar10', 'mnist', 'celeba64'])
    parser.add_argument('--data_path',  default='dataset')
    parser.add_argument('--n',          type=int, default=8,
                        help='Number of images to reconstruct (ignored when image files given)')
    parser.add_argument('--bn_iters',   type=int, default=50)
    parser.add_argument('--out_dir',    default='samples')
    parser.add_argument('imgs',         nargs='*',
                        help='Optional: paths to image files to reconstruct')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    H = cfg['model']['input_size']

    model = AutoEncoder(cfg['model']).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    epoch = ckpt['epoch']
    print(f'Loaded checkpoint (epoch {epoch})')

    # load images
    if args.imgs:
        originals = torch.cat([load_image(p, H, device) for p in args.imgs])
    else:
        originals = get_dataset_images(args.dataset, args.data_path, args.n, H, device)

    n = originals.size(0)
    print(f'Reconstructing {n} images...')

    print('Recalibrating BN stats...')
    model.train()
    with torch.no_grad():
        for _ in range(args.bn_iters):
            model(originals)
    model.eval()

    # encode → decode
    with torch.no_grad():
        z_list = model.encode(originals)
        recons = to_01(model.decode_from_z(z_list), model.decoder_type)

    orig_row = originals.cpu()
    if args.dataset == 'mnist':
        # grayscale → repeat to RGB for consistent display
        orig_row = orig_row.expand(-1, 3, -1, -1)

    rows = [orig_row, recons.cpu()]
    grid_imgs = torch.cat(rows, dim=0)           # (rows*n, C, H, W)
    grid = make_grid(grid_imgs, nrow=n, padding=2, normalize=False)

    # save grid (originals top, reconstructions bottom)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'e{epoch}_recon_n{n}.png')
    save_image(grid, path)
    print(f'Saved: {path}')
    print(f'  top row    : originals')
    print(f'  bottom row : reconstructions (encode → decode)')


if __name__ == '__main__':
    main()
