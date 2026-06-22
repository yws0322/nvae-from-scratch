"""Latent space exploration: visualize what each latent group controls.

For each of --n base images, saves:
  {out_dir}/e{epoch}_base_{i}.png        — base image i
  {out_dir}/e{epoch}_explore_{i}.png     — grid of all group perturbations for image i

Layout modes:
  default   : flat grid, nrow = groups_per_scale (e.g. 8x3 for 3-scale 8-group model)
  --per_scale: one row-block per scale; scale with N groups → ceil(N/nrow) rows of nrow.
               Shorter rows padded with grey. Fine scale (20 groups) → 2 rows of 10.

Usage:
    uv run python latent_space_exploration.py --ckpt ckpts/cifar10_8x3.pt --config configs/cifar10_8x3.yaml --n 2
    uv run python latent_space_exploration.py --ckpt ckpts/celeba64.pt --config configs/celeba64.yaml --per_scale --nrow 10
    uv run python latent_space_exploration.py --ckpt ... --groups 0 1 5 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import make_grid, save_image

from src.model import AutoEncoder
from src.utils import load_config, set_bn, _restore_bn_stats


def to_01(imgs, decoder_type):
    if decoder_type != 'bernoulli':
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    return imgs


def make_per_scale_grid(perturbed, gps, nrow=10, padding=2):
    """One row-block per scale, displayed coarse→fine (top→bottom).
    gps is fine→coarse order; we collect then reverse for display.
    Coarse scales (fewer groups than nrow) are padded with grey at the top.
    """
    # Decoder assigns groups coarse→fine; gps is fine→coarse, so reverse to slice correctly.
    scales = []
    idx = 0
    for n in reversed(gps):          # [5, 10, 20] coarse→fine
        scales.append(perturbed[idx:idx + n])
        idx += n

    rows = []
    for scale_imgs in scales:
        remainder = len(scale_imgs) % nrow
        if remainder:
            blank = torch.full_like(scale_imgs[0], 0.5)
            scale_imgs = scale_imgs + [blank] * (nrow - remainder)

        batch = torch.cat(scale_imgs, dim=0)          # (n_padded, C, H, W)
        row_grid = make_grid(batch, nrow=nrow, padding=padding)  # (C, H_row, W)
        rows.append(row_grid)

    return torch.cat(rows, dim=1)                     # (C, H_total, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      required=True)
    parser.add_argument('--config',    default='configs/cifar10.yaml')
    parser.add_argument('--t',         type=float, default=0.7,  help='Sampling temperature')
    parser.add_argument('--n',         type=int,   default=1,    help='Number of base images')
    parser.add_argument('--groups',    type=int,   nargs='*',    default=None,
                        help='Group indices to perturb (default: all)')
    parser.add_argument('--nrow',      type=int,   default=None,
                        help='Columns per row (default: 10 for per_scale, gps[0] otherwise)')
    parser.add_argument('--per_scale', action='store_true',
                        help='Layout one row-block per scale (fine split into rows of nrow)')
    parser.add_argument('--bn_iters',  type=int,   default=100)
    parser.add_argument('--out_dir',   default='samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    model = AutoEncoder(cfg['model']).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    epoch = ckpt['epoch']
    print(f'Loaded checkpoint (epoch {epoch})')

    total_groups = sum(model.gps)
    groups = args.groups if args.groups is not None else list(range(total_groups))

    if args.per_scale:
        nrow = args.nrow if args.nrow is not None else 10
        print(f'gps={model.gps} (coarse→fine)  |  per_scale layout  |  nrow={nrow}')
    else:
        nrow = args.nrow if args.nrow is not None else model.gps[0]
        print(f'Total latent groups: {total_groups}  |  Perturbing: {len(groups)}  |  nrow={nrow}')

    os.makedirs(args.out_dir, exist_ok=True)
    bn_saved = set_bn(model, device, t=args.t, num_samples=2, iters=args.bn_iters)

    # for each base image: perturb one group at a time, decode, save grid
    with torch.no_grad():
        for i in range(args.n):
            img_base, z_list = model.sample_with_z(1, device, t=args.t)
            img_base = to_01(img_base, model.decoder_type).cpu()
            save_image(img_base, os.path.join(args.out_dir, f'e{epoch}_base_{i}.png'), padding=0)

            perturbed = []
            for g in groups:
                img_g = to_01(model.explore_from_group(z_list, g, device, t=args.t),
                               model.decoder_type).cpu()
                perturbed.append(img_g)

            out_path = os.path.join(args.out_dir, f'e{epoch}_explore_{i}.png')
            if args.per_scale and args.groups is None:
                grid = make_per_scale_grid(perturbed, model.gps, nrow=nrow)
                save_image(grid.unsqueeze(0), out_path, nrow=1, padding=0)
            else:
                grid = torch.cat(perturbed, dim=0)  # (len(groups), C, H, W)
                save_image(grid, out_path, nrow=nrow, padding=2)
            print(f'  [{i}] base + explore saved')

    _restore_bn_stats(model, bn_saved)
    print(f'Done → {args.out_dir}/')


if __name__ == '__main__':
    main()
