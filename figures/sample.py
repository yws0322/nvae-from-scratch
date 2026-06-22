"""Sample from a trained NVAE checkpoint.

Usage:
    uv run python sample.py --ckpt ckpts/cifar10_8x3_aux_last.pt --config configs/cifar10_8x3_aux.yaml --dataset cifar10
    uv run python sample.py --ckpt ckpts/celeba64_aux_last.pt --config configs/celeba64_aux.yaml --dataset celeba64
    uv run python sample.py --ckpt runs/mnist_v2_kl_last.pt --config configs/mnist.yaml --dataset mnist
"""

import argparse
import os

import torch
from torchvision.utils import save_image

from src.model import AutoEncoder
from src.modules.distributions import DiscMixLogistic
from src.utils import load_config, set_bn, _restore_bn_stats
from train import get_dataset


@torch.no_grad()
def eval_val(model, val_loader, device, bpd_norm, dataset):
    model.eval()
    total_recon = total_kl = 0.0
    n = 0
    for x, _ in val_loader:
        x = x.to(device)
        if dataset == 'mnist':
            x = torch.bernoulli(x)
        out = model(x)
        logits, kl_all = out[0], out[1]  # forward may also return aux_out (3rd) on aux branch
        if dataset == 'mnist':
            recon = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, x, reduction='none').sum(dim=[1, 2, 3]).mean()
        else:
            recon = -DiscMixLogistic(logits).log_prob(2.0 * x - 1.0).sum(dim=[1, 2]).mean()
        kl = sum(k.mean() for k in kl_all)
        total_recon += recon.item()
        total_kl += kl.item()
        n += 1
    if dataset == 'mnist':
        print(f'Val NLL  (nats): {total_recon / n:.3f}')
        print(f'Val KL   (nats): {total_kl / n:.3f}')
        print(f'Val ELBO (nats): {(total_recon + total_kl) / n:.3f}')
    else:
        print(f'Val recon bpd : {total_recon / n / bpd_norm:.4f}')
        print(f'Val KL  (nats): {total_kl / n:.1f}')
        print(f'Val ELBO  bpd : {(total_recon / n + total_kl / n) / bpd_norm:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',      default='runs/cifar10_v2_new_last.pt')
    parser.add_argument('--config',    default='configs/cifar10_new.yaml')
    parser.add_argument('--dataset',   default='cifar10')
    parser.add_argument('--data_path', default='dataset')
    parser.add_argument('--t',         type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--n',         type=int, default=64,   help='Number of samples')
    parser.add_argument('--bn_iters',  type=int, default=500,  help='BN warmup iterations')
    parser.add_argument('--no_eval',   action='store_true',    help='Skip val bpd computation')
    parser.add_argument('--out_dir',   default='samples')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    model = AutoEncoder(cfg['model']).to(device)
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    result = model.load_state_dict(ckpt['model'], strict=False)
    if result.missing_keys:
        print(f'Missing keys: {result.missing_keys}')
    epoch = ckpt.get('epoch', '?')
    print(f'Loaded epoch {epoch}  |  groups: {sum(model.gps)}')

    H = W = cfg['model']['input_size']
    C = cfg['model']['input_channels']
    bpd_norm = H * W * C * 0.6931472

    # val ELBO
    if not args.no_eval:
        val_ds = get_dataset(args.dataset, args.data_path, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=100, num_workers=4, pin_memory=True)

        print('Running val eval...')
        eval_val(model, val_loader, device, bpd_norm, args.dataset)

    # BN recalibration + sample
    print(f'\nWarming up BN for t={args.t} ({args.bn_iters} iters)...')
    bn_saved = set_bn(model, device, t=args.t, num_samples=2, iters=args.bn_iters)

    print(f'Sampling {args.n} images...')
    with torch.no_grad():
        samples = model.sample(args.n, device, t=args.t)
    _restore_bn_stats(model, bn_saved)

    if args.dataset != 'mnist':
        samples = (samples.clamp(-1, 1) + 1) / 2
    # MNIST: model.sample() already returns sigmoid output in [0, 1]

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f'ep{epoch}_t{args.t:.2f}.png')
    save_image(samples, path, nrow=8, padding=2)
    print(f'Saved → {path}')


if __name__ == '__main__':
    main()
