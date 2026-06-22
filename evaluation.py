"""IW-NLL evaluation for NVAE checkpoints.

Computes ELBO and importance-weighted NLL (K samples) on the validation set.
Config is loaded from the checkpoint itself (ckpt['cfg']).

Usage:
    uv run python evaluation.py --ckpt runs/cifar10_last.pt
    uv run python evaluation.py --ckpt runs/celeba64_last.pt --dataset celeba64 --K 200
    uv run python evaluation.py --ckpt runs/mnist_last.pt --dataset mnist --K 200
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


def eval_ckpt(ckpt_path, dataset, K, iw_batch, data_path, device):
    # load model
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['cfg']
    model_cfg = cfg['model']

    C = model_cfg['input_channels']
    H = model_cfg['input_size']
    bpd_norm = H * H * C * math.log(2)
    use_bernoulli = (dataset == 'mnist')

    print(f'Checkpoint : {os.path.basename(ckpt_path)}  (epoch {ckpt["epoch"]})')
    print(f'Model      : scales={model_cfg["num_scales"]}, '
          f'groups={model_cfg["num_groups_per_scale"]}, '
          f'ch={model_cfg["initial_channels"]}, '
          f'nf={model_cfg.get("num_nf_cells", 0)}')

    model = AutoEncoder(model_cfg).to(device)
    result = model.load_state_dict(ckpt['model'], strict=False)
    if result.missing_keys:
        print(f'Missing keys: {result.missing_keys[:5]}')
    model.eval()

    val_ds = get_dataset(dataset, data_path, train=False)

    # ELBO (single-sample, fast)
    t0 = time.time()
    elbo_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=128, num_workers=4, pin_memory=True)
    total_recon = total_kl = 0.0
    n = 0
    with torch.no_grad():
        for x, _ in elbo_loader:
            x = x.to(device)
            if use_bernoulli:
                x = torch.bernoulli(x)
            out = model(x)
            logits, kl_all = out[0], out[1]
            if use_bernoulli:
                recon = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, x, reduction='none').sum(dim=[1, 2, 3]).mean()
            else:
                recon = -DiscMixLogistic(logits).log_prob(
                    2.0 * x - 1.0).sum(dim=[1, 2]).mean()
            kl = sum(k.mean() for k in kl_all)
            total_recon += recon.item()
            total_kl += kl.item()
            n += 1

    elbo_val = (total_recon / n + total_kl / n)
    if use_bernoulli:
        print(f'ELBO (nats): {elbo_val:.3f}  ({time.time()-t0:.0f}s)')
    else:
        print(f'ELBO (bpd) : {elbo_val / bpd_norm:.4f}  ({time.time()-t0:.0f}s)')

    # IW-NLL (K importance samples)
    t0 = time.time()
    iw_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=iw_batch, num_workers=4, pin_memory=True)
    total_iw_nll = 0.0
    n_images = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(iw_loader):
            x = x.to(device)
            if use_bernoulli:
                x = torch.bernoulli(x)
            B = x.size(0)
            x_rep = x.repeat_interleave(K, dim=0)

            out = model(x_rep, iw=True)
            logits, log_ratios = out[0], out[1]

            if use_bernoulli:
                recon = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, x_rep, reduction='none').sum(dim=[1, 2, 3])
            else:
                recon = -DiscMixLogistic(logits).log_prob(
                    2.0 * x_rep - 1.0).sum(dim=[1, 2])

            log_ratio_sum = sum(lr for lr in log_ratios)
            log_w = (-recon - log_ratio_sum).view(B, K)
            iw_nll = -(torch.logsumexp(log_w, dim=1) - math.log(K))
            total_iw_nll += iw_nll.sum().item()
            n_images += B

            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (len(iw_loader) - i - 1)
                running = total_iw_nll / n_images
                if not use_bernoulli:
                    running /= bpd_norm
                print(f'  IW [{i+1}/{len(iw_loader)}]  running={running:.4f}  eta={eta:.0f}s')

    iw_val = total_iw_nll / n_images
    if use_bernoulli:
        print(f'IW-NLL (nats, K={K}): {iw_val:.3f}  ({time.time()-t0:.0f}s)')
    else:
        print(f'IW-NLL (bpd,  K={K}): {iw_val / bpd_norm:.4f}  ({time.time()-t0:.0f}s)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     required=True,          help='checkpoint path')
    parser.add_argument('--dataset',  default='cifar10',
                        choices=['cifar10', 'celeba64', 'mnist'])
    parser.add_argument('--K',        type=int, default=200,  help='importance samples')
    parser.add_argument('--iw_batch', type=int, default=4,    help='images per IW forward pass')
    parser.add_argument('--data',     default='dataset')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  K={args.K}  iw_batch={args.iw_batch}\n')

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    eval_ckpt(args.ckpt, args.dataset, args.K, args.iw_batch, args.data, device)


if __name__ == '__main__':
    main()
