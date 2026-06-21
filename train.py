"""Training script for NVAE.

Usage (single GPU):
    python train.py --config configs/cifar10.yaml --dataset cifar10 --run_name my_run [--logging]

Usage (multi-GPU, DDP):
    torchrun --nproc_per_node=N train.py --config configs/cifar10.yaml --dataset cifar10 --run_name my_run [--logging]

Resume:
    python train.py ... --resume_from runs/my_run_last.pt
"""

import argparse
import contextlib
import io
import math
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import torchvision
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(__file__))
from src.model import AutoEncoder
from src.modules.distributions import DiscMixLogistic
from src.utils import (load_config, kl_balancer, kl_anneal_coeff, linear_warmup_cosine_decay,
                       build_free_bits, progressive_active_scales, progressive_group_coeff)


def setup_ddp():
    """Initialize DDP process group when launched via torchrun; no-op otherwise."""
    rank = int(os.environ.get('RANK', -1))
    if rank == -1:
        return False, 0, 1, 0
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return True, local_rank, world_size, rank


class CelebA64Dataset(Dataset):
    """CelebA cropped to 64×64.

    Expects: data_path/img_align_celeba/img_align_celeba/*.jpg
             data_path/list_eval_partition.csv  (partition: 0=train, 1=val, 2=test)
    """
    def __init__(self, data_path, split, transform):
        img_dir = os.path.join(data_path, 'img_align_celeba', 'img_align_celeba')
        partition_file = os.path.join(data_path, 'list_eval_partition.csv')

        df = pd.read_csv(partition_file)
        split_id = {'train': 0, 'val': 1, 'test': 2}[split]
        self.files = [
            os.path.join(img_dir, fname)
            for fname in df[df['partition'] == split_id]['image_id'].tolist()
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), 0  # dummy label


class CelebAHQ256Dataset(Dataset):
    """CelebA-HQ 256×256 from HuggingFace (korexyz/celeba-hq-256x256).

    Downloads parquet shards on first use; cached under data_path/celebahq256/.
    Train: 6 shards (~28k images). Val: 1 shard (~2k images).
    """
    _REPO = 'korexyz/celeba-hq-256x256'
    _TRAIN_SHARDS = [f'data/train-0000{i}-of-00006.parquet' for i in range(6)]
    _VAL_SHARDS   = ['data/validation-00000-of-00001.parquet']

    def __init__(self, data_path, split, transform):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        local_dir = os.path.join(data_path, 'celebahq256')
        shards = self._TRAIN_SHARDS if split == 'train' else self._VAL_SHARDS

        self.img_bytes = []
        for shard in shards:
            local_path = hf_hub_download(
                repo_id=self._REPO,
                filename=shard,
                repo_type='dataset',
                local_dir=local_dir,
            )
            tbl = pq.read_table(local_path, columns=['image'])
            for row in tbl.to_pydict()['image']:
                self.img_bytes.append(row['bytes'])

        self.transform = transform

    def __len__(self):
        return len(self.img_bytes)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.img_bytes[idx])).convert('RGB')
        return self.transform(img), 0


def get_dataset(name, data_path, train):
    if name == 'cifar10':
        tf = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()]) if train else T.ToTensor()
        return torchvision.datasets.CIFAR10(data_path, train=train, transform=tf, download=True)
    elif name == 'mnist':
        # Pad 28→32: 28 // 2^(prepost+scales-1) = 7 (odd), causing spatial misalignment on upsample.
        tf = T.Compose([T.Pad(2), T.ToTensor()])
        return torchvision.datasets.MNIST(data_path, train=train, transform=tf, download=True)
    elif name == 'celeba64':
        # 140px center-crop removes background (standard CelebA preprocessing).
        tf_train = T.Compose([T.CenterCrop(140), T.Resize(64), T.RandomHorizontalFlip(), T.ToTensor()])
        tf_val   = T.Compose([T.CenterCrop(140), T.Resize(64), T.ToTensor()])
        split = 'train' if train else 'val'
        return CelebA64Dataset(data_path, split, tf_train if train else tf_val)
    elif name == 'celebahq256':
        tf_train = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])
        tf_val   = T.ToTensor()
        split = 'train' if train else 'val'
        return CelebAHQ256Dataset(data_path, split, tf_train if train else tf_val)
    else:
        raise ValueError(f'Unknown dataset: {name}')


def train_epoch(model, raw_model, loader, optimizer, device,
                epoch, total_epochs, cfg_train, step, metric_norm,
                wandb_run=None, is_main=True, is_ddp=False, use_amp=False,
                decoder_type='disc_mix_logistic', metric_label='bpd',
                active_scales=None, group_coeff=None, free_bits=None):
    """Run one training epoch. step counts optimizer steps (not batches) for LR scheduling."""
    model.train()
    optimizer.zero_grad()

    accum_steps = cfg_train.get('accum_steps', 1)
    steps_per_epoch = max(len(loader) // accum_steps, 1)
    warmup_steps = cfg_train['warmup_epochs'] * steps_per_epoch
    total_steps  = total_epochs * steps_per_epoch

    aux_weight = cfg_train.get('aux_weight', 1.0)

    total_loss = total_recon = total_kl = total_aux = 0.0
    n_opt_steps = 0
    win_loss = win_recon = win_kl = win_sr = win_aux = 0.0
    win_count = 0

    for batch_idx, (x, _) in enumerate(loader):
        x = x.to(device)
        if decoder_type == 'bernoulli':
            x = torch.bernoulli(x)  # dynamic binarization: pixel → Bernoulli sample
        kl_coeff = kl_anneal_coeff(epoch, total_epochs,
                                   cfg_train['kl_anneal_portion'],
                                   cfg_train['kl_const_portion'])

        is_last_accum = (batch_idx + 1) % accum_steps == 0
        is_last_batch = (batch_idx + 1) == len(loader)
        should_step   = is_last_accum or is_last_batch

        sync_ctx = contextlib.nullcontext() if (should_step or not is_ddp) else model.no_sync()

        with sync_ctx:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits, kl_all, aux_outs = model(x, active_scales=active_scales)
                # Progressive growing: active_scales < num_scales → logits are at the
                # active scale's coarse resolution (e.g. 8×8), use MSE against avg-pooled x.
                # Full mode: active_scales == num_scales → normal DiscMixLogistic / Bernoulli.
                prog_growing = (active_scales is not None
                                and active_scales < raw_model.num_scales
                                and raw_model.use_progressive)
                if prog_growing:
                    x_low = F.adaptive_avg_pool2d(x, logits.shape[2:])
                    recon_loss = ((torch.sigmoid(logits.float()) - x_low) ** 2
                                 ).sum(dim=[1, 2, 3]).mean()
                elif decoder_type == 'bernoulli':
                    recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, x, reduction='none').sum(dim=[1, 2, 3]).mean()
                else:
                    output_dist = DiscMixLogistic(logits)
                    recon_loss  = -output_dist.log_prob(2.0 * x - 1.0).sum(dim=[1, 2]).mean()
                # Slice to active groups only; in full mode this is a no-op.
                n_active = len(kl_all)
                kl_loss  = kl_balancer(kl_all, kl_coeff, kl_balance=(kl_coeff < 1.0),
                                       kl_alpha=raw_model.kl_alpha[:n_active],
                                       free_bits=(free_bits[:n_active]
                                                  if free_bits is not None else None),
                                       group_coeff=(group_coeff[:n_active]
                                                    if group_coeff is not None else None))
                aux_loss = torch.zeros((), device=device)
                if aux_outs:
                    for i, a in enumerate(aux_outs):
                        w = aux_weight[i] if isinstance(aux_weight, (list, tuple)) else aux_weight
                        x_low = F.adaptive_avg_pool2d(x, a.shape[2:])
                        aux_loss = aux_loss + w * ((torch.sigmoid(a.float()) - x_low) ** 2
                                                   ).sum(dim=[1, 2, 3]).mean()
            # SR/BN in FP32: power iteration overflows in FP16.
            sr_loss     = cfg_train['weight_decay_norm'] * raw_model.spectral_norm_loss(active_scales=active_scales)
            bn_loss_val = cfg_train['bn_weight_decay'] * raw_model.bn_loss(active_scales=active_scales)
            loss        = recon_loss + kl_loss + sr_loss + bn_loss_val + aux_loss

            (loss / accum_steps).backward()

        if torch.isnan(loss):
            nan_recon = torch.isnan(recon_loss).item()
            nan_kl    = torch.isnan(kl_loss).item()
            nan_sr    = torch.isnan(sr_loss).item()
            nan_bn    = torch.isnan(bn_loss_val).item()
            grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                            if p.grad is not None) ** 0.5
            print(f'[NaN] epoch={epoch} batch={batch_idx} '
                  f'recon={nan_recon} kl={nan_kl} sr={nan_sr} bn={nan_bn} '
                  f'grad_norm={grad_norm:.1f} '
                  f'recon_val={recon_loss.item():.1f} kl_val={kl_loss.item():.1f} '
                  f'sr_val={sr_loss.item():.4f}', flush=True)
            optimizer.zero_grad()
            break

        win_loss  += loss.item()
        win_recon += recon_loss.item()
        win_kl    += kl_loss.item()
        win_sr    += sr_loss.item()
        if aux_outs:
            win_aux += aux_loss.item()
        win_count += 1

        if should_step:
            nn.utils.clip_grad_norm_(model.parameters(), cfg_train['grad_clip'])
            optimizer.step()
            optimizer.zero_grad()

            lr = linear_warmup_cosine_decay(optimizer, step, warmup_steps, total_steps,
                                            cfg_train['base_lr'],
                                            lr_min=cfg_train.get('lr_min', 0.0))
            step += 1
            n_opt_steps += 1

            avg_loss  = win_loss  / win_count
            avg_recon = win_recon / win_count
            avg_kl    = win_kl    / win_count
            avg_sr    = win_sr    / win_count
            avg_aux   = win_aux   / win_count

            total_loss  += avg_loss
            total_recon += avg_recon
            total_kl    += avg_kl
            if aux_outs:
                total_aux += avg_aux

            if is_main and wandb_run is not None and step % 50 == 0:
                log_dict = {
                    'train/loss':              avg_loss,
                    'train/recon':             avg_recon,
                    f'train/{metric_label}':   avg_recon / metric_norm,
                    'train/kl':                avg_kl,
                    'train/sr':                avg_sr,
                    'train/kl_coeff': kl_coeff,
                    'train/lr':       lr,
                }
                if raw_model.use_aux_recon:
                    log_dict['train/aux'] = avg_aux
                wandb_run.log(log_dict, step=step)

            win_loss = win_recon = win_kl = win_sr = win_aux = 0.0
            win_count = 0

    n = max(n_opt_steps, 1)
    return total_loss / n, total_recon / n, total_kl / n, total_aux / n, step


@torch.no_grad()
def eval_epoch(model, loader, device, decoder_type='disc_mix_logistic', num_samples=1,
               active_scales=None, prog_growing=False):
    """Evaluate on validation set. num_samples > 1 reduces KL variance via averaging."""
    model.eval()
    total_recon = 0.0
    total_kl    = 0.0
    n_batches = 0
    kl_per_group = None

    for x, _ in loader:
        x = x.to(device)
        if decoder_type == 'bernoulli':
            x = torch.bernoulli(x)  # dynamic binarization: match training distribution
        logits, kl_all, _ = model(x, active_scales=active_scales)
        if prog_growing:
            x_low = F.adaptive_avg_pool2d(x, logits.shape[2:])
            recon_loss = ((torch.sigmoid(logits.float()) - x_low) ** 2
                         ).sum(dim=[1, 2, 3]).mean()
        elif decoder_type == 'bernoulli':
            recon_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, x, reduction='none').sum(dim=[1, 2, 3]).mean()
        else:
            output_dist = DiscMixLogistic(logits)
            recon_loss = -output_dist.log_prob(2.0 * x - 1.0).sum(dim=[1, 2]).mean()
        total_recon += recon_loss.item()

        kl_samples = [k.mean().item() for k in kl_all]
        for _ in range(num_samples - 1):
            _, kl_extra, _ = model(x, active_scales=active_scales)
            kl_samples = [a + b.mean().item() for a, b in zip(kl_samples, kl_extra)]
        total_kl += sum(v / num_samples for v in kl_samples)

        batch_kls = [v / num_samples for v in kl_samples]
        if kl_per_group is None:
            kl_per_group = batch_kls
        else:
            kl_per_group = [a + b for a, b in zip(kl_per_group, batch_kls)]

        n_batches += 1

    if dist.is_initialized():
        stats = torch.tensor([total_recon, total_kl, float(n_batches)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_recon, total_kl, n_batches = stats[0].item(), stats[1].item(), int(stats[2].item())

        if kl_per_group:
            kl_t = torch.tensor(kl_per_group, device=device)
            dist.all_reduce(kl_t, op=dist.ReduceOp.SUM)
            kl_per_group = kl_t.tolist()

    n = max(n_batches, 1)
    kl_per_group = [v / n for v in kl_per_group] if kl_per_group else []
    return total_recon / n, total_kl / n, kl_per_group


def main():
    is_ddp, local_rank, world_size, rank = setup_ddp()
    is_main = (rank <= 0)  # rank 0 owns logging, printing, and checkpointing

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          default='configs/cifar10.yaml')
    parser.add_argument('--data_path',       default='dataset')
    parser.add_argument('--dataset',         default='cifar10')
    parser.add_argument('--run_name',        default='run')
    parser.add_argument('--checkpoint_dir',  default='./runs')
    parser.add_argument('--resume_from',     default=None)
    parser.add_argument('--logging',         action='store_true', help='Enable WandB logging')
    parser.add_argument('--wandb_project',   default='nvae-from-scratch')
    parser.add_argument('--wandb_id',        default=None, help='WandB run ID to resume (overrides checkpoint)')
    parser.add_argument('--batch_size',      type=int, default=None,
                        help='Per-GPU batch size; overrides config value')
    parser.add_argument('--nf_cells',        type=int, default=None,
                        help='NFBlocks per latent group (0=off, 2=paper default); overrides config')
    parser.add_argument('--accum_steps',     type=int, default=None,
                        help='Gradient accumulation steps; effective bs = bs * world_size * accum_steps')
    parser.add_argument('--base_lr',         type=float, default=None,
                        help='Base learning rate; overrides config value')
    parser.add_argument('--amp',             action='store_true',
                        help='Enable automatic mixed precision (float16 forward, float32 grads)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_model = cfg['model']
    cfg_train = cfg['training']
    decoder_type = 'bernoulli' if args.dataset == 'mnist' else 'disc_mix_logistic'
    cfg_model['decoder'] = decoder_type
    if args.batch_size  is not None: cfg_train['batch_size']      = args.batch_size
    if args.nf_cells    is not None: cfg_model['num_nf_cells']    = args.nf_cells
    if args.accum_steps is not None: cfg_train['accum_steps']     = args.accum_steps
    if args.base_lr     is not None: cfg_train['base_lr']         = args.base_lr
    cfg_train.setdefault('accum_steps', 1)

    use_amp = args.amp and torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if is_main:
        accum = cfg_train['accum_steps']
        eff_bs = cfg_train['batch_size'] * world_size * accum
        print(f'Device: {device} | world_size: {world_size} | '
              f'per-GPU bs: {cfg_train["batch_size"]} | accum: {accum} | effective bs: {eff_bs} | '
              f'AMP: {use_amp}')

    wandb_run = None
    if is_main and args.logging:
        import wandb
        _wandb_id = args.run_name
        if args.wandb_id:
            _wandb_id = args.wandb_id
        elif args.resume_from and os.path.exists(args.resume_from):
            _ckpt_peek = torch.load(args.resume_from, map_location='cpu', weights_only=False)
            _wandb_id = _ckpt_peek.get('wandb_id', args.run_name)
            del _ckpt_peek
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            id=_wandb_id,
            resume='allow',
            config={
                'model':                cfg_model,
                'training':             cfg_train,
                'dataset':              args.dataset,
                'world_size':           world_size,
                'effective_batch_size': cfg_train['batch_size'] * world_size * cfg_train['accum_steps'],
                'amp':                  use_amp,
            },
        )

    train_ds = get_dataset(args.dataset, args.data_path, train=True)
    val_ds   = get_dataset(args.dataset, args.data_path, train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True)  if is_ddp else None
    val_sampler   = DistributedSampler(val_ds,   shuffle=False) if is_ddp else None

    num_workers = min(os.cpu_count() or 4, 8)
    train_loader = DataLoader(
        train_ds, batch_size=cfg_train['batch_size'],
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg_train['batch_size'],
        sampler=val_sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )

    use_progressive = bool(cfg_train.get('progressive', False))
    prog_boundaries = list(cfg_train.get('progressive_epochs', []))
    prog_kl_warmup  = cfg_train.get('progressive_kl_warmup', 5)
    num_scales      = cfg_model['num_scales']
    if use_progressive:
        expected = num_scales - 1
        if len(prog_boundaries) != expected:
            raise ValueError(
                f'progressive_epochs must have exactly num_scales-1={expected} entries '
                f'(one unlock boundary per non-coarsest scale), got {prog_boundaries}'
            )
        if not all(a < b for a, b in zip(prog_boundaries, prog_boundaries[1:])):
            raise ValueError(
                f'progressive_epochs must be strictly increasing (no duplicates), '
                f'got {prog_boundaries}'
            )
        total_epochs = cfg_train['epochs']
        if any(b <= 0 or b >= total_epochs for b in prog_boundaries):
            raise ValueError(
                f'All progressive_epochs must be in (0, {total_epochs}), got {prog_boundaries}'
            )
        if is_main:
            print(f'Progressive-scale training: boundaries={prog_boundaries} '
                  f'(num_scales={num_scales}), kl_warmup={prog_kl_warmup} epochs')

    # propagate progressive flag to model config so AutoEncoder builds prog_heads.
    cfg_model['progressive'] = use_progressive

    model = AutoEncoder(cfg_model).to(device)
    if is_ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=use_progressive)
    raw_model = model.module if is_ddp else model

    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        print(f'Parameters: {n_params:,}')

    # weight_norm _v/_g and prior_ftr0 must not have L2 decay:
    # decaying _v/g destabilises weight_norm; decaying prior_ftr0 pushes it to ~0.
    wd = cfg_train.get('weight_decay', 0.0)
    _no_wd = lambda n: n.endswith('_v') or n.endswith('_g') or n.endswith('prior_ftr0')
    decay_params    = [p for n, p in model.named_parameters() if not _no_wd(n)]
    no_decay_params = [p for n, p in model.named_parameters() if     _no_wd(n)]
    optimizer = torch.optim.Adamax(
        [{'params': decay_params, 'weight_decay': wd},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=cfg_train['base_lr'], eps=1e-3,
    )

    start_epoch = 0
    step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location=device)
        result = raw_model.load_state_dict(ckpt['model'], strict=False)
        if is_main and result.missing_keys:
            print(f'Checkpoint missing keys: {result.missing_keys}')
        if is_main and result.unexpected_keys:
            print(f'Checkpoint unexpected keys: {result.unexpected_keys}')
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        step        = ckpt.get('step', 0)
        if is_main:
            print(f'Resumed from epoch {start_epoch}')

    if is_main:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    total_epochs = cfg_train['epochs']
    H = W = cfg_model['input_size']
    C = cfg_model['input_channels']
    if decoder_type == 'bernoulli':
        metric_norm  = 1.0
        metric_label = 'nll'
    else:
        metric_norm  = H * W * C * 0.6931472   # nats → bpd
        metric_label = 'bpd'

    # Free bits: per-group KL floor (Proposal 1). Scalar or per-scale list in config.
    free_bits_cfg = cfg_train.get('free_bits', None)
    free_bits = (build_free_bits(raw_model.gps, num_scales, free_bits_cfg)
                 if free_bits_cfg is not None else None)

    for epoch in range(start_epoch, total_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if use_progressive:
            active_scales = progressive_active_scales(epoch, prog_boundaries, num_scales)
            group_coeff = progressive_group_coeff(
                epoch, prog_boundaries, num_scales, raw_model.group_scale,
                prog_kl_warmup, device)
        else:
            active_scales = None
            group_coeff = None

        loss, recon, kl, aux, step = train_epoch(
            model, raw_model, train_loader, optimizer, device,
            epoch, total_epochs, cfg_train, step, metric_norm,
            wandb_run, is_main=is_main, is_ddp=is_ddp, use_amp=use_amp,
            decoder_type=decoder_type, metric_label=metric_label,
            active_scales=active_scales, group_coeff=group_coeff, free_bits=free_bits,
        )
        prog_growing_epoch = (use_progressive and active_scales is not None
                              and active_scales < num_scales)
        val_recon, val_kl, kl_per_group = eval_epoch(
            model, val_loader, device, decoder_type=decoder_type, num_samples=4,
            active_scales=active_scales, prog_growing=prog_growing_epoch)
        val_metric      = val_recon / metric_norm
        val_metric_elbo = (val_recon + val_kl) / metric_norm

        if is_main:
            train_metric = recon / metric_norm
            prog_tag = f' active={active_scales}/{num_scales}' if use_progressive else ''
            print(f'Epoch {epoch:03d} | loss={loss:.3f} recon={recon:.3f} kl={kl:.3f} '
                  f'train_{metric_label}={train_metric:.3f} val_{metric_label}={val_metric:.3f} '
                  f'val_kl={val_kl:.3f} val_{metric_label}_elbo={val_metric_elbo:.3f}{prog_tag}')

        if is_main and wandb_run is not None:
            import wandb
            log_dict = {
                'epoch':                             epoch,
                'train/epoch_loss':                  loss,
                'train/epoch_recon':                 recon,
                f'train/epoch_{metric_label}':       recon / metric_norm,
                'train/epoch_kl':                    kl,
                'val/recon':                         val_recon,
                'val/kl':                            val_kl,
                f'val/{metric_label}':               val_metric,
                f'val/{metric_label}_elbo':          val_metric_elbo,
            }
            if use_progressive:
                log_dict['train/active_scales'] = active_scales
            for i, kl_g in enumerate(kl_per_group):
                log_dict[f'val/kl_group_{i:03d}'] = kl_g

            wandb_run.log(log_dict, step=step)

        if is_main and not (math.isnan(loss) or math.isnan(val_recon)) and \
                ((epoch + 1) % 10 == 0 or epoch == total_epochs - 1):
            ckpt_path = os.path.join(args.checkpoint_dir, f'{args.run_name}_last.pt')
            ckpt = {
                'epoch':     epoch,
                'step':      step,
                'model':     raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'cfg':       cfg,
                'wandb_id':  wandb_run.id if wandb_run is not None else None,
            }
            torch.save(ckpt, ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')

    if wandb_run is not None:
        wandb_run.finish()

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
