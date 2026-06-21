"""IW-bpd evaluation for CIFAR-10 checkpoints.

Usage:
    uv run python evaluation.py [--ckpt PATH] [--K 500] [--iw_batch 4] [--data dataset]

If --ckpt is omitted, evaluates all CIFAR-10 checkpoints in /home/elicer/ckpts/.
Config is loaded from the checkpoint itself (ckpt['cfg']).
"""

import argparse
import math
import os
import time

import torch
import torch.utils.data

from src.model import AutoEncoder
from src.modules.distributions import DiscMixLogistic
from train import get_dataset

BPD_NORM = 32 * 32 * 3 * math.log(2)

DEFAULT_CKPTS = [
    '/home/elicer/ckpts/cifar10_official.pt',
    '/home/elicer/ckpts/cifar_official.pt',
    '/home/elicer/ckpts/cifar10_1x30.pt',
    '/home/elicer/ckpts/cifar10_15x2.pt',
    '/home/elicer/ckpts/cifar_8x3.pt',
]


def eval_ckpt(ckpt_path, K, iw_batch, data_path, device):
    print(f'\n{"="*60}')
    print(f'Checkpoint: {os.path.basename(ckpt_path)}')

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model_cfg = cfg['model']

    print(f'  epoch={ckpt["epoch"]}, scales={model_cfg["num_scales"]}, '
          f'groups={model_cfg["num_groups_per_scale"]}, ch={model_cfg["initial_channels"]}, '
          f'nf={model_cfg.get("num_nf_cells", 0)}')

    model = AutoEncoder(model_cfg).to(device)
    result = model.load_state_dict(ckpt['model'], strict=False)
    if result.missing_keys:
        print(f'  Missing keys: {result.missing_keys[:5]}')
    model.eval()

    val_ds = get_dataset('cifar10', data_path, train=False)

    # --- ELBO (fast) ---
    t0 = time.time()
    elbo_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, num_workers=4, pin_memory=True)
    total_recon = total_kl = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in elbo_loader:
            x = x.to(device)
            out = model(x)
            logits, kl_all = out[0], out[1]  # forward may also return aux_out (3rd) on aux branch
            recon = -DiscMixLogistic(logits).log_prob(
                2.0 * x - 1.0).sum(dim=[1, 2]).mean()
            kl = sum(k.mean() for k in kl_all)
            total_recon += recon.item()
            total_kl += kl.item()
            n += 1
    elbo_bpd = (total_recon / n + total_kl / n) / BPD_NORM
    print(f'  ELBO bpd : {elbo_bpd:.4f}  ({time.time()-t0:.0f}s)')

    # --- IW-bpd ---
    t0 = time.time()
    iw_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=iw_batch, num_workers=4, pin_memory=True)
    total_iw_nll = 0.0
    n_images = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(iw_loader):
            x = x.to(device)
            B = x.size(0)
            x_rep = x.repeat_interleave(K, dim=0)  # (B*K, C, H, W)

            out = model(x_rep, iw=True)
            logits, log_ratios = out[0], out[1]  # forward may also return aux_out (3rd) on aux branch

            recon = -DiscMixLogistic(logits).log_prob(
                2.0 * x_rep - 1.0).sum(dim=[1, 2])          # (B*K,)
            log_ratio_sum = sum(lr for lr in log_ratios)     # (B*K,)

            log_w = (-recon - log_ratio_sum).view(B, K)
            iw_nll = -(torch.logsumexp(log_w, dim=1) - math.log(K))  # (B,)
            total_iw_nll += iw_nll.sum().item()
            n_images += B

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(iw_loader) - i - 1)
                print(f'  IW [{i+1}/{len(iw_loader)}] '
                      f'running={total_iw_nll/n_images/BPD_NORM:.4f}  eta={eta:.0f}s')

    iw_bpd = total_iw_nll / n_images / BPD_NORM
    print(f'  IW-bpd (K={K}): {iw_bpd:.4f}  ({time.time()-t0:.0f}s)')
    print(f'  (paper target: 2.91 w/ NF, 2.93 w/o NF)')
    return elbo_bpd, iw_bpd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=None, help='single checkpoint path; omit to eval all')
    parser.add_argument('--K', type=int, default=500, help='IW samples')
    parser.add_argument('--iw_batch', type=int, default=4, help='images per IW forward pass')
    parser.add_argument('--data', default='dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}, K={args.K}, iw_batch={args.iw_batch}')

    ckpts = [args.ckpt] if args.ckpt else DEFAULT_CKPTS

    results = []
    for ckpt_path in ckpts:
        if not os.path.exists(ckpt_path):
            print(f'Skipping (not found): {ckpt_path}')
            continue
        elbo, iw = eval_ckpt(ckpt_path, args.K, args.iw_batch, args.data, device)
        results.append((os.path.basename(ckpt_path), elbo, iw))

    print(f'\n{"="*60}')
    print('Summary:')
    print(f'  {"checkpoint":<30} {"ELBO":>8} {"IW-bpd":>8}')
    for name, elbo, iw in results:
        print(f'  {name:<30} {elbo:>8.4f} {iw:>8.4f}')


if __name__ == '__main__':
    main()
