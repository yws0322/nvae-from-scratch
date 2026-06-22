import contextlib
import yaml
import torch


def _coerce_floats(obj):
    # PyYAML parses scientific notation (e.g. 1e-3) as a string, not a float.
    if isinstance(obj, dict):
        return {k: _coerce_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_floats(v) for v in obj]
    if isinstance(obj, str):
        try:
            return int(obj)
        except ValueError:
            pass
        try:
            return float(obj)
        except ValueError:
            pass
    return obj


def load_config(path):
    with open(path) as f:
        return _coerce_floats(yaml.safe_load(f))


def build_free_bits(gps, num_scales, per_scale):
    """Build per-group KL floor tensor for free-bits training (Proposal 1).

    per_scale: scalar or list of length num_scales, coarse→fine order.
    Returns (total_groups,) tensor in decoder coarse→fine group order.
    """
    if isinstance(per_scale, (int, float)):
        per_scale = [float(per_scale)] * num_scales
    assert len(per_scale) == num_scales, \
        f'free_bits must be scalar or len={num_scales}, got len={len(per_scale)}'
    fb = []
    for s_idx in range(num_scales):
        enc_s = num_scales - 1 - s_idx   # decoder processes coarse→fine
        fb.extend([float(per_scale[s_idx])] * gps[enc_s])
    return torch.tensor(fb)


def kl_balancer(kl_all, kl_coeff, kl_balance=True, kl_alpha=None,
                free_bits=None, group_coeff=None):
    """Weighted KL loss across all latent groups.

    free_bits: (G,) KL floor per group. Use floored KL for balancer weights so a
    collapsed group (KL≈0) still gets a meaningful gradient proportional to floor/mean_kl.
    group_coeff: per-group multiplier for progressive training; normalized by Σ gc (not
    n_active) to prevent a newly-unlocked scale from inflating already-active groups' weights.
    """
    kl_vals = torch.stack(kl_all, dim=1)   # (B, G)
    kl_for_loss = kl_vals if free_bits is None \
        else torch.maximum(kl_vals, free_bits.to(kl_vals.device).unsqueeze(0))

    if kl_balance:
        kl_coeff_i = kl_for_loss.abs().mean(dim=0, keepdim=True) + 0.01  # (1, G)
        total_kl = kl_coeff_i.sum()
        if kl_alpha is not None:
            kl_coeff_i = kl_coeff_i / kl_alpha.unsqueeze(0) * total_kl
        if group_coeff is not None:
            kl_coeff_i = kl_coeff_i * group_coeff.unsqueeze(0)
            coeff_sum = group_coeff.sum().clamp(min=1e-8)
            effective_mean = kl_coeff_i.sum(dim=1, keepdim=True) / coeff_sum
            kl_coeff_i = kl_coeff_i / (effective_mean + 1e-8)
        else:
            kl_coeff_i = kl_coeff_i / kl_coeff_i.mean(dim=1, keepdim=True)
        kl_loss = (kl_for_loss * kl_coeff_i.detach()).sum(dim=1).mean()
    elif group_coeff is not None:
        kl_loss = (kl_for_loss * group_coeff.unsqueeze(0)).sum(dim=1).mean()
    else:
        kl_loss = kl_for_loss.sum(dim=1).mean()

    return kl_coeff * kl_loss


def progressive_active_scales(epoch, boundaries, num_scales):
    """Active scale count at this epoch: starts at 1, increments at each boundary epoch."""
    active = 1 + sum(1 for b in boundaries if epoch >= b)
    return min(active, num_scales)


def progressive_group_coeff(epoch, boundaries, num_scales, group_scale,
                            warmup_epochs, device, min_coeff=1e-4):
    """Per-group KL multiplier: 0 for locked scales, linearly ramped for recently-unlocked, 1 otherwise."""
    unlock_epoch = [0] + list(boundaries)
    active = progressive_active_scales(epoch, boundaries, num_scales)
    coeffs = []
    for s in group_scale:
        if s >= active:
            coeffs.append(0.0)
        elif warmup_epochs and warmup_epochs > 0 and unlock_epoch[s] > 0:
            frac = (epoch - unlock_epoch[s]) / warmup_epochs
            coeffs.append(min(1.0, max(min_coeff, frac)))
        else:
            coeffs.append(1.0)
    return torch.tensor(coeffs, device=device)


def kl_anneal_coeff(epoch, total_epochs, anneal_portion, const_portion, min_kl_coeff=1e-4):
    """KL annealing coefficient in [min_kl_coeff, 1]. Floor prevents encoder from fully ignoring prior."""
    anneal_start = const_portion * total_epochs
    anneal_end = (const_portion + anneal_portion) * total_epochs
    if epoch < anneal_start:
        return min_kl_coeff
    elif epoch < anneal_end:
        raw = (epoch - anneal_start) / (anneal_end - anneal_start)
        return max(raw, min_kl_coeff)
    else:
        return 1.0


def _save_bn_stats(model):
    import torch.nn as nn
    _bn = (nn.BatchNorm2d, nn.SyncBatchNorm)
    return {name: (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
            for name, m in model.named_modules() if isinstance(m, _bn)}


def _restore_bn_stats(model, stats):
    import torch.nn as nn
    _bn = (nn.BatchNorm2d, nn.SyncBatchNorm)
    for name, m in model.named_modules():
        if isinstance(m, _bn) and name in stats:
            mean, var, nbt = stats[name]
            m.running_mean.copy_(mean)
            m.running_var.copy_(var)
            m.num_batches_tracked.copy_(nbt)


def set_bn(model, device, t=1.0, num_samples=2, iters=100, amp=False):
    """Recalibrate BN running stats for sampling at temperature t; returns saved training stats."""
    saved = _save_bn_stats(model)
    model.train()
    amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
               if amp else contextlib.nullcontext())
    with torch.no_grad(), amp_ctx:
        for _ in range(iters):
            model.sample(num_samples, device, t=t)
    model.eval()
    return saved


def linear_warmup_cosine_decay(optimizer, step, warmup_steps, total_steps, base_lr, lr_min=0.0):
    """Update optimizer LR: linear warmup then cosine decay to lr_min."""
    import math
    if step < warmup_steps:
        lr = base_lr * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = lr_min + (base_lr - lr_min) * 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr
