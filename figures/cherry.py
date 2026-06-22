"""Interactive cherry-picking: sample and curate the best generated images.

Each round samples --grid_n images, saves the grid to a file, and asks you to
enter the 1-based indices of images you want to keep. Repeat until --winners
images are collected, then a final selection round lets you narrow them down.

No GUI is required: just open the saved PNG file in any image viewer.

Usage:
    uv run python cherry.py --ckpt runs/cifar10_v1_last.pt
    uv run python cherry.py --ckpt runs/... --t 0.7 --winners 9 --grid_n 9
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision.utils import save_image

from src.model import AutoEncoder
from src.utils import load_config, set_bn, _restore_bn_stats


def to_01(imgs, decoder_type):
    if decoder_type != 'bernoulli':
        imgs = (imgs.clamp(-1, 1) + 1) / 2
    return imgs


def parse_indices(line, n):
    """Parse space-separated 1-based indices from user input. Returns list of 0-based indices."""
    result = []
    for tok in line.strip().split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < n:
                result.append(idx)
            else:
                print(f'  ! {tok} out of range (1–{n}), skipped')
        except ValueError:
            print(f'  ! "{tok}" is not a number, skipped')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     required=True)
    parser.add_argument('--config',   default='configs/cifar10.yaml')
    parser.add_argument('--t',        type=float, default=0.7)
    parser.add_argument('--winners',  type=int,   default=9,
                        help='Number of images to collect before the final round')
    parser.add_argument('--grid_n',   type=int,   default=9,
                        help='Images shown per sampling round')
    parser.add_argument('--bn_iters', type=int,   default=100)
    parser.add_argument('--out_dir',  default='samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    model = AutoEncoder(cfg['model']).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    epoch = ckpt['epoch']
    print(f'Loaded checkpoint (epoch {epoch})')

    os.makedirs(args.out_dir, exist_ok=True)
    bn_saved = set_bn(model, device, t=args.t, num_samples=2, iters=args.bn_iters)

    nrow_batch = max(1, int(args.grid_n ** 0.5))
    nrow_final = max(1, int(args.winners ** 0.5))
    selected = []
    round_num = 0

    print(f'\nCollecting {args.winners} images, {args.grid_n} shown per round.')
    print('Open the saved PNG, enter space-separated indices (1-based) to keep, Enter to skip.\n')

    while len(selected) < args.winners:
        round_num += 1
        with torch.no_grad():
            imgs = to_01(model.sample(args.grid_n, device, t=args.t), model.decoder_type).cpu()

        batch_path = os.path.join(args.out_dir, f'cherry_round_{round_num:03d}.png')
        save_image(imgs, batch_path, nrow=nrow_batch, padding=2)
        print(f'Round {round_num} — have {len(selected)}/{args.winners}')
        print(f'  View: {batch_path}')
        print(f'  Pick (1–{args.grid_n}): ', end='', flush=True)

        line = input()
        for idx in parse_indices(line, args.grid_n):
            selected.append(imgs[idx])
            print(f'    + image {idx+1} added ({len(selected)}/{args.winners})')

    _restore_bn_stats(model, bn_saved)

    # Final selection round
    selected_t = torch.stack(selected[:args.winners])
    final_path = os.path.join(args.out_dir, f'cherry_candidates_e{epoch}.png')
    save_image(selected_t, final_path, nrow=nrow_final, padding=2)
    print(f'\nAll {len(selected_t)} candidates saved: {final_path}')
    print(f'Enter subset indices to narrow down (e.g. "1 3 7"), or Enter to keep all: ', end='', flush=True)
    line = input()

    if line.strip():
        idxs = parse_indices(line, len(selected_t))
        if idxs:
            selected_t = torch.stack([selected_t[i] for i in idxs])

    out_path = os.path.join(args.out_dir, f'cherry_final_e{epoch}_t{args.t:.2f}.png')
    save_image(selected_t, out_path, nrow=nrow_final, padding=2)
    print(f'Final {len(selected_t)} images saved: {out_path}')


if __name__ == '__main__':
    main()
