"""Latent space interpolation between two images.

Encodes two images to their posterior latent codes, then linearly interpolates
between the code lists and decodes each step. The output grid shows:

  [image A] [interp 1] [interp 2] ... [interp N] [image B]

Usage:
    uv run python latent_interpolation.py --ckpt runs/cifar10_v1_last.pt img1.png img2.png
    uv run python latent_interpolation.py --ckpt runs/... img1.png img2.png --steps 8
"""

import argparse
import os

import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from src.model import AutoEncoder
from src.utils import load_config


def load_image(path, size, device):
    img = Image.open(path).convert('RGB')
    img = TF.resize(img, [size, size])
    return TF.to_tensor(img).unsqueeze(0).to(device)   # (1, 3, H, W) in [0, 1]


def to_01(imgs, decoder_type):
    if decoder_type != 'bernoulli':
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    return imgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     required=True)
    parser.add_argument('--config',   default='configs/cifar10.yaml')
    parser.add_argument('img1')
    parser.add_argument('img2')
    parser.add_argument('--steps',    type=int, default=6,
                        help='Number of interpolation steps between the two images')
    parser.add_argument('--bn_iters', type=int, default=50)
    parser.add_argument('--out_dir',  default='samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    H = cfg['model']['input_size']

    model = AutoEncoder(cfg['model']).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    epoch = ckpt['epoch']
    print(f'Loaded checkpoint (epoch {epoch})')

    img_a = load_image(args.img1, H, device)
    img_b = load_image(args.img2, H, device)

    # Recalibrate BN running stats in train mode before encoding.
    print('Recalibrating BN stats...')
    model.train()
    with torch.no_grad():
        for _ in range(args.bn_iters):
            model(img_a)
            model(img_b)
    model.eval()

    with torch.no_grad():
        z_a = model.encode(img_a)
        z_b = model.encode(img_b)

        frames = [img_a.cpu()]
        for i in range(1, args.steps + 1):
            alpha = i / (args.steps + 1)
            z_interp = [(1.0 - alpha) * za + alpha * zb for za, zb in zip(z_a, z_b)]
            img_interp = to_01(model.decode_from_z(z_interp), model.decoder_type)
            frames.append(img_interp.cpu())
        frames.append(img_b.cpu())

    os.makedirs(args.out_dir, exist_ok=True)
    grid = torch.cat(frames, dim=0)   # (steps+2, C, H, W)
    name_a = os.path.splitext(os.path.basename(args.img1))[0]
    name_b = os.path.splitext(os.path.basename(args.img2))[0]
    path = os.path.join(args.out_dir, f'e{epoch}_interp_{name_a}_to_{name_b}_s{args.steps}.png')
    save_image(grid, path, nrow=len(frames), padding=2)
    print(f'Saved: {path}')
    print(f'  {len(frames)} frames: [img_a] + {args.steps} interpolations + [img_b]')


if __name__ == '__main__':
    main()
