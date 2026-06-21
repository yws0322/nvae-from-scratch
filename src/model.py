"""NVAE: hierarchical VAE with residual posteriors, depthwise-separable decoder, spectral regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.architecture import (
    ResidualCellEncoder, ResidualCellDecoder,
    EncCombinerCell, DecCombinerCell, NFChain,
)
from src.modules.distributions import Normal, DiscMixLogistic

CHANNEL_MULT = 2  # channels double at each scale/preprocess transition


def _effective_weight(layer):
    # weight_norm only refreshes layer.weight inside forward(); a never-forwarded layer
    # (locked sampler in progressive training) keeps a stale, possibly CPU-bound .weight.

    from torch.nn.utils.weight_norm import WeightNorm
    for hook in layer._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm):
            return hook.compute_weight(layer)
    return layer.weight


def groups_per_scale(num_scales, num_groups, is_adaptive, min_groups=1):
    # gps[0] = finest scale (most groups); gps[-1] = coarsest.
    g, n = [], num_groups
    for _ in range(num_scales):
        g.append(n)
        if is_adaptive:
            n = max(min_groups, n // 2)
    return g


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        C0 = cfg['initial_channels']
        C_in = cfg.get('input_channels', 3)
        self.num_scales = cfg['num_scales']
        self.num_prepost_blocks = cfg['num_prepost_blocks']
        self.num_prepost_cells = cfg['num_prepost_cells']
        self.num_cells_per_group = cfg['num_cells_per_group']
        self.C_z = cfg['num_latent_per_group']
        self.num_mix = cfg.get('num_logistic_mixtures', 10)
        self.use_se = cfg.get('use_se', True)
        self.res_dist = cfg.get('res_dist', True)
        self.decoder_type = cfg.get('decoder', 'disc_mix_logistic')

        self.gps = groups_per_scale(
            self.num_scales,
            cfg['num_groups_per_scale'],
            cfg.get('is_adaptive', False),
            cfg.get('min_groups_per_scale', 1),
        )
        self.num_nf_cells = cfg.get('num_nf_cells', 0)

        self.stem = nn.Conv2d(C_in, C0, 3, padding=1, bias=True)

        self.pre_blocks = nn.ModuleList()
        mult = 1
        for _ in range(self.num_prepost_blocks):
            block = nn.ModuleList()
            for c in range(self.num_prepost_cells):
                C_i = int(C0 * mult)
                if c == self.num_prepost_cells - 1:          # last cell: downsample
                    C_o = int(C0 * mult * CHANNEL_MULT)
                    block.append(ResidualCellEncoder(C_i, C_o, stride=2, use_se=self.use_se))
                    mult *= CHANNEL_MULT
                else:
                    block.append(ResidualCellEncoder(C_i, C_i, stride=1, use_se=self.use_se))
            self.pre_blocks.append(block)

        self.enc_cells = nn.ModuleList()
        self.enc_combiners = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        for s in range(self.num_scales):
            C = int(C0 * mult)
            for g in range(self.gps[s]):
                group_cells = nn.ModuleList(
                    [ResidualCellEncoder(C, C, stride=1, use_se=self.use_se)
                     for _ in range(self.num_cells_per_group)]
                )
                self.enc_cells.append(group_cells)
                is_last = (s == self.num_scales - 1) and (g == self.gps[s] - 1)
                if not is_last:
                    self.enc_combiners.append(EncCombinerCell(C, C))
            if s < self.num_scales - 1:
                C_next = int(C * CHANNEL_MULT)
                self.enc_downs.append(ResidualCellEncoder(C, C_next, stride=2, use_se=self.use_se))
                mult *= CHANNEL_MULT

        C_bot = int(C0 * mult)
        self.enc_bottleneck = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(C_bot, C_bot, 1, bias=True),
            nn.SiLU(),
        )

        self.enc_samplers = nn.ModuleList()
        self.dec_samplers = nn.ModuleList()

        dec_mult = mult
        for s in range(self.num_scales):
            enc_s = self.num_scales - 1 - s
            C = int(C0 * dec_mult)
            for g in range(self.gps[enc_s]):
                self.enc_samplers.append(
                    nn.utils.weight_norm(nn.Conv2d(C, 2 * self.C_z, 3, padding=1, bias=True))
                )
                if not (s == 0 and g == 0):
                    self.dec_samplers.append(
                        nn.Sequential(
                            nn.SiLU(),
                            nn.utils.weight_norm(nn.Conv2d(C, 2 * self.C_z, 1, bias=True)),
                        )
                    )
            dec_mult /= CHANNEL_MULT

        # group_scale[g]: decoder scale index (0=coarsest) for global group g.
        self.group_scale = []
        for s in range(self.num_scales):
            enc_s = self.num_scales - 1 - s
            self.group_scale.extend([s] * self.gps[enc_s])

        # Map decoder group g → its EncCombinerCell for SR/BN exclusion during progressive training.
        # DDP marks locked modules "ready" before backward; computing SR/BN loss through them
        # fires their gradient hook twice → "marked ready twice" crash.
        # enc_combiners are saved finest-first then reversed: decoder group g uses enc_combiners[G-1-g].
        G = sum(self.gps)
        self._group_enc_combiner = [None]
        for g in range(1, G):
            self._group_enc_combiner.append(self.enc_combiners[G - 1 - g])

        C_ftr0 = int(C0 * mult)
        spatial_scale = 2 ** (self.num_prepost_blocks + self.num_scales - 1)
        H_ftr0 = cfg['input_size'] // spatial_scale
        self.prior_ftr0 = nn.Parameter(torch.rand(C_ftr0, H_ftr0, H_ftr0) * 0.01)

        self.dec_cells = nn.ModuleList()
        self.dec_combiners = nn.ModuleList()
        self.dec_ups = nn.ModuleList()

        dec_mult = mult
        for s in range(self.num_scales):
            enc_s = self.num_scales - 1 - s
            C = int(C0 * dec_mult)
            for g in range(self.gps[enc_s]):
                is_first = (s == 0 and g == 0)
                if not is_first:
                    group_cells = nn.ModuleList(
                        [ResidualCellDecoder(C, expansion=6, use_se=self.use_se)
                         for _ in range(self.num_cells_per_group)]
                    )
                    self.dec_cells.append(group_cells)
                self.dec_combiners.append(DecCombinerCell(C, self.C_z))
            if s < self.num_scales - 1:
                C_next = int(C / CHANNEL_MULT)
                self.dec_ups.append(nn.Conv2d(C, C_next, 1, bias=True))
                dec_mult /= CHANNEL_MULT

        self.post_blocks = nn.ModuleList()
        post_mult = dec_mult
        for _ in range(self.num_prepost_blocks):
            block = nn.ModuleList()
            for c in range(self.num_prepost_cells):
                C_i = int(C0 * post_mult)
                if c == 0:
                    C_o = int(C_i / CHANNEL_MULT)
                    block.append(nn.Conv2d(C_i, C_o, 1, bias=True))
                    post_mult /= CHANNEL_MULT
                else:
                    C_o = int(C0 * post_mult)
                    block.append(ResidualCellDecoder(C_o, expansion=3, use_se=self.use_se))
            self.post_blocks.append(block)

        # ----------------------------------------------------------- image head
        C_out_head = int(C0 * post_mult)
        if self.decoder_type == 'bernoulli':
            self.image_head = nn.Sequential(
                nn.ELU(),
                nn.Conv2d(C_out_head, 1, 3, padding=1, bias=True),
            )
        else:
            self.image_head = nn.Sequential(
                nn.ELU(),
                nn.Conv2d(C_out_head, 10 * self.num_mix, 3, padding=1, bias=True),
            )

        # prog_heads[k-1]: output head used when only k < num_scales scales are active.
        # MSE against avg-pooled x replaces DiscMixLogistic/Bernoulli during growing stages.
        self.use_progressive = cfg.get('progressive', False)
        if self.use_progressive and self.num_scales > 1:
            out_ch = 1 if self.decoder_type == 'bernoulli' else C_in
            self.prog_heads = nn.ModuleList()
            ph_mult = mult
            for _ in range(self.num_scales - 1):
                C_ph = int(C0 * ph_mult)
                self.prog_heads.append(nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(C_ph, out_ch, 3, padding=1, bias=True),
                ))
                ph_mult /= CHANNEL_MULT

        # aux_heads: one per coarse scale; reconstruct avg-pooled x to give coarse z's a direct
        # pixel signal that finer groups cannot absorb (Proposal 2, aux_recon=true in config).
        self.use_aux_recon = cfg.get('aux_recon', False)
        if self.use_aux_recon:
            self.n_aux = min(int(cfg.get('aux_scales', 1)), self.num_scales - 1)
            if self.n_aux == 0:
                raise ValueError(
                    f"aux_recon=True requires num_scales >= 2 (got {self.num_scales})"
                )
            out_ch = 1 if self.decoder_type == 'bernoulli' else C_in
            self.aux_heads = nn.ModuleList()
            aux_mult = mult
            for _ in range(self.n_aux):
                C_s = int(C0 * aux_mult)
                self.aux_heads.append(nn.Sequential(
                    nn.SiLU(),
                    nn.Conv2d(C_s, out_ch, 3, padding=1, bias=True),
                ))
                aux_mult /= CHANNEL_MULT

        # kl_alpha: per-group KL weight proportional to spatial area (finer = larger weight, mean=1).
        self.register_buffer('kl_alpha', self._compute_kl_alpha())

        # Spectral regularization: track left/right singular vectors per conv layer.
        # Exclude aux_heads and prog_heads — they're small prediction heads, not part of the encoder.
        self.all_conv_layers = []
        self.sr_u = {}
        self.sr_v = {}
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d) and 'aux_head' not in name and 'prog_head' not in name:
                self.all_conv_layers.append(layer)
        self._init_sr_buffers()

        self._bn_types = (nn.BatchNorm2d, nn.SyncBatchNorm)

        total_groups = sum(self.gps)
        if self.num_nf_cells > 0:
            self.nf_blocks = nn.ModuleList([
                NFChain(self.C_z, self.num_nf_cells)
                for _ in range(total_groups)
            ])

    def _compute_kl_alpha(self):
        # Weight = (2^s_idx)^2 / groups_at_scale, normalized so min=1 (matches official NVAE).
        weights = []
        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            n_groups = self.gps[enc_s]
            w = float((2 ** s_idx) ** 2) / n_groups
            weights.extend([w] * n_groups)
        t = torch.tensor(weights)
        return t / t.min()

    NUM_POWER_ITER = 4

    def _init_sr_buffers(self, num_init_iters=10):
        for layer in self.all_conv_layers:
            W = _effective_weight(layer)
            h = W.view(W.size(0), -1)
            with torch.no_grad():
                u = F.normalize(torch.randn(h.size(0), device=W.device), dim=0)
                for _ in range(num_init_iters * self.NUM_POWER_ITER):
                    v = F.normalize(h.t() @ u, dim=0)
                    u = F.normalize(h @ v, dim=0)
            self.sr_u[id(layer)] = u.detach()
            self.sr_v[id(layer)] = v.detach()

    def _locked_module_ids(self, active_scales):
        """Module ids absent from the forward graph during progressive growing.

        SR/BN losses must skip these — DDP marks locked params "ready" before backward,
        so a second gradient through them fires the hook twice → "marked ready twice" crash.
        """
        ids: set = set()
        if active_scales is None or active_scales >= self.num_scales:
            return ids

        def add(module):
            if module is not None:
                ids.update(id(m) for m in module.modules())

        for g, s in enumerate(self.group_scale):
            if s < active_scales:
                continue
            add(self.enc_samplers[g])
            add(self._group_enc_combiner[g])
            if self.use_progressive:
                add(self.dec_combiners[g])
                if g > 0:
                    add(self.dec_cells[g - 1])
                    add(self.dec_samplers[g - 1])
        if self.use_progressive:
            for s_up in range(active_scales - 1, len(self.dec_ups)):
                add(self.dec_ups[s_up])
            add(self.post_blocks)
            add(self.image_head)
        return ids

    def spectral_norm_loss(self, active_scales=None):
        """Sum of max singular values across active conv layers (power iteration)."""
        locked_ids = self._locked_module_ids(active_scales)

        total = 0.0
        for layer in self.all_conv_layers:
            if id(layer) in locked_ids:
                continue
            W = _effective_weight(layer)
            h = W.view(W.size(0), -1)
            u = self.sr_u[id(layer)].to(W.device)
            v = self.sr_v[id(layer)].to(W.device)
            with torch.no_grad():
                for _ in range(self.NUM_POWER_ITER):
                    v_new = F.normalize(h.t() @ u, dim=0)
                    u_new = F.normalize(h @ v_new, dim=0)
                    u, v = u_new, v_new
                self.sr_u[id(layer)] = u_new.detach()
                self.sr_v[id(layer)] = v_new.detach()
            sigma = (u_new @ h @ v_new)
            total = total + sigma
        return total

    def bn_loss(self, active_scales=None):
        """Sum of max BN scale params across active BN layers."""
        locked_ids = self._locked_module_ids(active_scales)
        total = 0.0
        for m in self.modules():
            if isinstance(m, self._bn_types) and m.weight is not None:
                if id(m) in locked_ids:
                    continue
                total = total + m.weight.abs().max()
        return total

    def forward(self, x, iw=False, active_scales=None):
        """Returns (logits, kl_all, aux_outs). active_scales=k locks finer scales to prior (progressive training)."""
        if active_scales is None:
            active_scales = self.num_scales
        s = self.stem(2.0 * x - 1.0)

        for block in self.pre_blocks:
            for cell in block:
                s = cell(s)

        # Bottom-up encoder: save features + combiner refs for decoder use.
        # EncCombinerCells are NOT applied here — they also need the decoder state.
        saved_enc = []
        saved_comb = []
        comb_idx = 0
        enc_cell_idx = 0
        down_idx = 0

        for s_idx in range(self.num_scales):
            for g in range(self.gps[s_idx]):
                for cell in self.enc_cells[enc_cell_idx]:
                    s = cell(s)
                enc_cell_idx += 1
                is_last = (s_idx == self.num_scales - 1) and (g == self.gps[s_idx] - 1)
                if not is_last:
                    saved_enc.append(s)
                    saved_comb.append(self.enc_combiners[comb_idx])
                    comb_idx += 1
            if s_idx < self.num_scales - 1:
                s = self.enc_downs[down_idx](s)
                down_idx += 1

        ftr = self.enc_bottleneck(s)

        # Top-down decoder: iterate coarsest-to-finest; encoder features consumed in reverse.
        saved_enc.reverse()
        saved_comb.reverse()

        batch = ftr.size(0)
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch, -1, -1, -1)

        kl_all = []
        aux_outs = [] if self.use_aux_recon else None
        enc_idx = dec_cell_idx = dec_samp_idx = up_idx = global_group = 0

        # prog_growing: stop at active_scales, use prog_heads output, skip post+image_head.
        prog_growing = self.use_progressive and active_scales < self.num_scales
        loop_scales = active_scales if prog_growing else self.num_scales

        for s_idx in range(loop_scales):
            enc_s = self.num_scales - 1 - s_idx
            # scale_locked: non-progressive path only; prog_growing already limits loop_scales.
            scale_locked = (not prog_growing) and (s_idx >= active_scales)

            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)

                if is_first:
                    # First group: prior = N(0, I)
                    param_q = self.enc_samplers[0](ftr)
                    dmu, dlog_sig = param_q.chunk(2, dim=1)
                    q = Normal(dmu, dlog_sig)
                    p = Normal(torch.zeros_like(dmu), torch.zeros_like(dlog_sig))
                else:
                    param_p = self.dec_samplers[dec_samp_idx](dec_state)
                    mu_p, lsig_p = param_p.chunk(2, dim=1)
                    p = Normal(mu_p, lsig_p)
                    dec_samp_idx += 1

                    if scale_locked:
                        # Locked group (non-progressive path): posterior collapses to the
                        # prior. z is drawn from p (no encoder signal), KL = 0 exactly.
                        q = p
                    else:
                        # Posterior (residual parameterization)
                        enc_feat = saved_enc[enc_idx]
                        combiner = saved_comb[enc_idx]
                        enc_idx += 1
                        ftr_combined = combiner(enc_feat, dec_state)
                        param_q = self.enc_samplers[global_group](ftr_combined)
                        dmu, dlog_sig = param_q.chunk(2, dim=1)
                        if self.res_dist:
                            q = Normal(mu_p + dmu, lsig_p + dlog_sig)
                        else:
                            q = Normal(dmu, dlog_sig)

                # Sample z and compute KL
                z_q, _ = q.sample()
                if scale_locked:
                    z = z_q
                    kl = dec_state.new_zeros(z_q.size(0))
                elif self.num_nf_cells > 0:
                    # NF change-of-variables: KL = log q(z_q) − log_det − log p(z_nf)
                    z, log_det = self.nf_blocks[global_group](z_q)
                    kl = (q.log_prob(z_q).sum(dim=[1, 2, 3])
                          - log_det
                          - p.log_prob(z).sum(dim=[1, 2, 3]))
                else:
                    z = z_q
                    if iw:
                        kl = (q.log_prob(z_q) - p.log_prob(z_q)).sum(dim=[1, 2, 3])
                    else:
                        kl = q.kl(p).sum(dim=[1, 2, 3])
                kl_all.append(kl)

                # dec_combiner before dec_cells: z conditions the residual computation.
                dec_state = self.dec_combiners[global_group](dec_state, z)
                global_group += 1

                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if self.use_aux_recon and s_idx < self.n_aux:
                aux_outs.append(self.aux_heads[s_idx](dec_state))

            if s_idx < loop_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        if prog_growing:
            logits = self.prog_heads[active_scales - 1](dec_state)
        else:
            for block in self.post_blocks:
                for c_idx, cell in enumerate(block):
                    if c_idx == 0:
                        dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                        dec_state = cell(dec_state)
                    else:
                        dec_state = cell(dec_state)
            logits = self.image_head(dec_state)

        return logits, kl_all, aux_outs

    @torch.no_grad()
    def sample(self, batch_size, device, t=1.0):
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        dec_samp_idx = dec_cell_idx = up_idx = global_group = 0

        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)
                if is_first:
                    p = Normal(
                        torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                        torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                    ).scale_temperature(t)
                else:
                    param_p = self.dec_samplers[dec_samp_idx](dec_state)
                    mu_p, lsig_p = param_p.chunk(2, dim=1)
                    p = Normal(mu_p, lsig_p).scale_temperature(t)
                    dec_samp_idx += 1

                z_p, _ = p.sample()
                z = z_p  # p is prior on z_nf space; no NF transform during generation
                dec_state = self.dec_combiners[global_group](dec_state, z)
                global_group += 1

                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if s_idx < self.num_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        for block in self.post_blocks:
            for c_idx, cell in enumerate(block):
                if c_idx == 0:
                    dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                    dec_state = cell(dec_state)
                else:
                    dec_state = cell(dec_state)

        logits = self.image_head(dec_state)
        if self.decoder_type == 'bernoulli':
            return torch.sigmoid(logits)
        return DiscMixLogistic(logits, num_bits=8).sample(t=1.0)

    @torch.no_grad()
    def explore_from_group(self, z_prefix, start_group, device, t=1.0):
        """Fix z_{0..start_group-1}; resample z_{start_group..L-1} from learned priors.

        Avoids OOD z's from naive N(0,t) replacement by using conditioned prior samples.
        """
        batch_size = z_prefix[0].size(0)
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        dec_cell_idx = up_idx = global_group = 0

        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)

                if global_group < start_group:
                    z = z_prefix[global_group]
                else:
                    if is_first:
                        p = Normal(
                            torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                            torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                        ).scale_temperature(t)
                    else:
                        param_p = self.dec_samplers[global_group - 1](dec_state)
                        mu_p, lsig_p = param_p.chunk(2, dim=1)
                        p = Normal(mu_p, lsig_p).scale_temperature(t)
                    z, _ = p.sample()

                dec_state = self.dec_combiners[global_group](dec_state, z)
                global_group += 1
                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if s_idx < self.num_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        for block in self.post_blocks:
            for c_idx, cell in enumerate(block):
                if c_idx == 0:
                    dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                    dec_state = cell(dec_state)
                else:
                    dec_state = cell(dec_state)

        logits = self.image_head(dec_state)
        if self.decoder_type == 'bernoulli':
            return torch.sigmoid(logits)
        return DiscMixLogistic(logits, num_bits=8).sample(t=1.0)

    @torch.no_grad()
    def encode(self, x):
        """Encode x to posterior latent codes. Returns z_list (coarse-to-fine)."""
        s = self.stem(2.0 * x - 1.0)
        for block in self.pre_blocks:
            for cell in block:
                s = cell(s)

        saved_enc, saved_comb = [], []
        comb_idx = enc_cell_idx = down_idx = 0
        for s_idx in range(self.num_scales):
            for g in range(self.gps[s_idx]):
                for cell in self.enc_cells[enc_cell_idx]:
                    s = cell(s)
                enc_cell_idx += 1
                is_last = (s_idx == self.num_scales - 1) and (g == self.gps[s_idx] - 1)
                if not is_last:
                    saved_enc.append(s)
                    saved_comb.append(self.enc_combiners[comb_idx])
                    comb_idx += 1
            if s_idx < self.num_scales - 1:
                s = self.enc_downs[down_idx](s)
                down_idx += 1

        ftr = self.enc_bottleneck(s)
        saved_enc.reverse()
        saved_comb.reverse()

        batch = ftr.size(0)
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch, -1, -1, -1)
        z_list = []
        enc_idx = dec_samp_idx = dec_cell_idx = up_idx = global_group = 0

        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)
                if is_first:
                    param_q = self.enc_samplers[0](ftr)
                    dmu, dlog_sig = param_q.chunk(2, dim=1)
                    q = Normal(dmu, dlog_sig)
                else:
                    param_p = self.dec_samplers[dec_samp_idx](dec_state)
                    mu_p, lsig_p = param_p.chunk(2, dim=1)
                    dec_samp_idx += 1
                    ftr_combined = saved_comb[enc_idx](saved_enc[enc_idx], dec_state)
                    enc_idx += 1
                    param_q = self.enc_samplers[global_group](ftr_combined)
                    dmu, dlog_sig = param_q.chunk(2, dim=1)
                    q = Normal(mu_p + dmu, lsig_p + dlog_sig) if self.res_dist else Normal(dmu, dlog_sig)

                z_q, _ = q.sample()
                z = self.nf_blocks[global_group](z_q)[0] if self.num_nf_cells > 0 else z_q
                z_list.append(z)

                dec_state = self.dec_combiners[global_group](dec_state, z)
                global_group += 1
                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if s_idx < self.num_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        return z_list

    @torch.no_grad()
    def decode_from_z(self, z_list):
        batch_size = z_list[0].size(0)
        device = z_list[0].device
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        dec_cell_idx = up_idx = global_group = 0

        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)
                dec_state = self.dec_combiners[global_group](dec_state, z_list[global_group])
                global_group += 1
                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if s_idx < self.num_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        for block in self.post_blocks:
            for c_idx, cell in enumerate(block):
                if c_idx == 0:
                    dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                    dec_state = cell(dec_state)
                else:
                    dec_state = cell(dec_state)

        logits = self.image_head(dec_state)
        if self.decoder_type == 'bernoulli':
            return torch.sigmoid(logits)
        return DiscMixLogistic(logits, num_bits=8).sample(t=1.0)

    @torch.no_grad()
    def sample_with_z(self, batch_size, device, t=1.0):
        """Sample from prior; return (images, z_list) capturing the z values used."""
        dec_state = self.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        dec_samp_idx = dec_cell_idx = up_idx = global_group = 0
        z_list = []

        for s_idx in range(self.num_scales):
            enc_s = self.num_scales - 1 - s_idx
            for g in range(self.gps[enc_s]):
                is_first = (s_idx == 0 and g == 0)
                if is_first:
                    p = Normal(
                        torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                        torch.zeros(batch_size, self.C_z, *dec_state.shape[2:], device=device),
                    ).scale_temperature(t)
                else:
                    param_p = self.dec_samplers[dec_samp_idx](dec_state)
                    mu_p, lsig_p = param_p.chunk(2, dim=1)
                    p = Normal(mu_p, lsig_p).scale_temperature(t)
                    dec_samp_idx += 1

                z, _ = p.sample()
                z_list.append(z)
                dec_state = self.dec_combiners[global_group](dec_state, z)
                global_group += 1
                if not is_first:
                    for cell in self.dec_cells[dec_cell_idx]:
                        dec_state = cell(dec_state)
                    dec_cell_idx += 1

            if s_idx < self.num_scales - 1:
                dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                dec_state = self.dec_ups[up_idx](dec_state)
                up_idx += 1

        for block in self.post_blocks:
            for c_idx, cell in enumerate(block):
                if c_idx == 0:
                    dec_state = F.interpolate(dec_state, scale_factor=2, mode='bilinear', align_corners=False)
                    dec_state = cell(dec_state)
                else:
                    dec_state = cell(dec_state)

        logits = self.image_head(dec_state)
        if self.decoder_type == 'bernoulli':
            imgs = torch.sigmoid(logits)
        else:
            imgs = DiscMixLogistic(logits, num_bits=8).sample(t=1.0)
        return imgs, z_list
