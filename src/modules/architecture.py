import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-and-Excitation channel gating."""

    def __init__(self, C, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(C, max(C // reduction, 4)),
            nn.SiLU(),
            nn.Linear(max(C // reduction, 4), C),
            nn.Sigmoid(),
        )

    def forward(self, x):
        gate = x.mean(dim=[2, 3])
        gate = self.fc(gate).unsqueeze(-1).unsqueeze(-1)
        return x * gate


class ResidualCellEncoder(nn.Module):
    """BN-Swish-Conv3x3-BN-Swish-Conv3x3 + optional SE. Skip uses Conv1x1 on shape change."""

    def __init__(self, C_in, C_out, stride=1, use_se=True):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(C_in, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(C_in, C_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(C_out, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(C_out, C_out, 3, padding=1, bias=False),
        )
        self.se = SE(C_out) if use_se else nn.Identity()
        if stride > 1 or C_in != C_out:
            self.skip = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.skip(x) + 0.1 * self.se(self.residual(x))


class ResidualCellDecoder(nn.Module):
    """BN-Conv1x1(expand×6)-BN-Swish-DWConv5x5-BN-Swish-Conv1x1-BN + optional SE. Upsample done outside."""

    def __init__(self, C, expansion=6, use_se=True):
        super().__init__()
        C_mid = C * expansion
        self.residual = nn.Sequential(
            nn.BatchNorm2d(C, momentum=0.05),
            nn.Conv2d(C, C_mid, 1, bias=False),
            nn.BatchNorm2d(C_mid, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(C_mid, C_mid, 5, padding=2, groups=C_mid, bias=False),  # depthwise
            nn.BatchNorm2d(C_mid, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(C_mid, C, 1, bias=False),
            nn.BatchNorm2d(C, momentum=0.05),
        )
        self.se = SE(C) if use_se else nn.Identity()

    def forward(self, x):
        return x + 0.1 * self.se(self.residual(x))


class EncCombinerCell(nn.Module):
    """Fuse encoder and decoder features for posterior: h_enc + Conv1x1(h_dec)."""

    cell_type = 'combiner_enc'

    def __init__(self, C_enc, C_dec):
        super().__init__()
        self.proj = nn.Conv2d(C_dec, C_enc, 1, bias=True)

    def forward(self, h_enc, h_dec):
        return h_enc + self.proj(h_dec)


class DecCombinerCell(nn.Module):
    """Inject sampled z into decoder stream: Conv1x1(concat([h, z]))."""

    cell_type = 'combiner_dec'

    def __init__(self, C_dec, C_z):
        super().__init__()
        self.proj = nn.Conv2d(C_dec + C_z, C_dec, 1, bias=True)

    def forward(self, h, z):
        return self.proj(torch.cat([h, z], dim=1))


class MaskedConv2d(nn.Conv2d):
    """Autoregressive conv with raster-scan spatial ordering + lower-triangular channel ordering."""

    def __init__(self, C_in, C_out, kernel_size=1, groups=1, zero_diag=True, **kwargs):
        if kernel_size > 1 and 'padding' not in kwargs:
            kwargs['padding'] = kernel_size // 2
        super().__init__(C_in, C_out, kernel_size, groups=groups, **kwargs)
        mask = self._build_mask(C_in, C_out, kernel_size, groups, zero_diag)
        self.register_buffer('mask', mask)

    @staticmethod
    def _channel_mask(C_in, C_out, zero_diag):
        """Lower-triangular channel mask supporting expansion (C_out >= C_in) and compression (C_out <= C_in)."""
        mask = torch.zeros(C_out, C_in)
        if C_out >= C_in:
            assert C_out % C_in == 0, f"C_out {C_out} must be divisible by C_in {C_in}"
            r = C_out // C_in
            for i in range(C_in):
                if i > 0:
                    mask[i * r:(i + 1) * r, :i] = 1   # see all past groups
                if not zero_diag:
                    mask[i * r:(i + 1) * r, i] = 1    # also see own group
        else:
            assert C_in % C_out == 0, f"C_in {C_in} must be divisible by C_out {C_out}"
            r = C_in // C_out
            for i in range(C_out):
                if i > 0:
                    mask[i, :i * r] = 1
                if not zero_diag:
                    mask[i, i * r:(i + 1) * r] = 1
        return mask

    @staticmethod
    def _build_mask(C_in, C_out, k, groups, zero_diag):
        m = k // 2
        if groups > 1:
            # Depthwise: each output channel sees only its own input channel
            assert groups == C_in and C_out == C_in, \
                "depthwise expects groups == C_in == C_out"
            mask = torch.ones(C_out, 1, k, k)
            if k > 1:
                mask[:, :, m + 1:, :] = 0
                mask[:, :, m, m + 1:] = 0
                if zero_diag:
                    mask[:, :, m, m] = 0
            return mask

        ch = MaskedConv2d._channel_mask(C_in, C_out, zero_diag)
        if k == 1:
            return ch.view(C_out, C_in, 1, 1)

        # k > 1, non-depthwise: spatial raster-scan + channel ordering at center
        mask = torch.ones(C_out, C_in, k, k)
        mask[:, :, m + 1:, :] = 0        # rows after center → future
        mask[:, :, m, m + 1:] = 0        # right of center in center row → future
        mask[:, :, m, m] = ch            # center pixel: channel-dependent
        return mask

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def _mix_log_cdf_flow(z, logit_pi, mu, log_s, log_a, b):
    """Mixture-of-logistics CDF transform with analytic log-determinant."""
    log_s = torch.clamp(log_s, min=-7.0)
    z_    = z.unsqueeze(2)                                          # (B,C,1,H,W)

    log_pi  = F.log_softmax(logit_pi, dim=2)                       # (B,C,K,H,W)
    u       = -(z_ - mu) * torch.exp(-log_s)                       # (B,C,K,H,W)
    sp_u    = F.softplus(u)                                        # log(1+exp(u))

    log_cdf       = torch.logsumexp(log_pi - sp_u,      dim=2)    # (B,C,H,W)
    log_one_minus = torch.logsumexp(log_pi + u - sp_u,  dim=2)    # (B,C,H,W)

    log_a_ = log_a.squeeze(2)                                      # (B,C,H,W)
    b_     = b.squeeze(2)
    z_new  = torch.exp(log_a_) * (log_cdf - log_one_minus) + b_

    # log |d(z_new_i)/d(z_i)| = log_a + log_pdf − log_cdf − log(1−cdf)
    log_pdf = torch.logsumexp(log_pi + u - log_s - 2 * sp_u, dim=2)
    log_det = (log_a_ - log_cdf - log_one_minus + log_pdf).sum(dim=[1, 2, 3])

    return z_new, log_det


class NFCell(nn.Module):
    """AR flow step: MaskedConv3×3(C→6C) → MaskedDWConv5×5 → MaskedConv1×1 → MixLogCDF transform."""

    NUM_MIX = 3

    def __init__(self, C_z, reverse=False, expansion=6):
        super().__init__()
        self.reverse = reverse
        self.C_z = C_z
        C_h = C_z * expansion
        K   = self.NUM_MIX

        self.ar = nn.Sequential(
            MaskedConv2d(C_z, C_h, kernel_size=3, groups=1,   zero_diag=True,  bias=True),
            nn.ELU(inplace=True),
            MaskedConv2d(C_h, C_h, kernel_size=5, groups=C_h, zero_diag=False, bias=True),
            nn.ELU(inplace=True),
        )
        # 3K+3 params per channel: (logit_pi×K, mu×K, log_s×K, log_a, b, dummy)
        self.params = MaskedConv2d(C_h, C_z * (3 * K + 3), kernel_size=1,
                                   groups=1, zero_diag=False, bias=True)
        nn.init.zeros_(self.params.weight)
        nn.init.zeros_(self.params.bias)

    def forward(self, z):
        x      = z.flip(1) if self.reverse else z
        h      = self.ar(x)
        raw    = self.params(h)                          # (B, C_z*(3K+3), H, W)

        B, _, H, W = raw.shape
        K   = self.NUM_MIX
        raw = raw.view(B, self.C_z, 3 * K + 3, H, W)
        logit_pi = raw[:, :, :K]
        mu       = raw[:, :, K:2 * K]
        log_s    = raw[:, :, 2 * K:3 * K]
        log_a    = raw[:, :, 3 * K:3 * K + 1]
        b        = raw[:, :, 3 * K + 1:3 * K + 2]

        z_new, log_det = _mix_log_cdf_flow(x, logit_pi, mu, log_s, log_a, b)
        if self.reverse:
            z_new = z_new.flip(1)
        return z_new, log_det


class NFBlock(nn.Module):
    """Pair of NFCells: forward (lower-triangular AR) + reversed (upper-triangular AR)."""

    def __init__(self, C_z):
        super().__init__()
        self.cell1 = NFCell(C_z, reverse=False)
        self.cell2 = NFCell(C_z, reverse=True)

    def forward(self, z):
        z,  ld1 = self.cell1(z)
        z,  ld2 = self.cell2(z)
        return z, ld1 + ld2


class NFChain(nn.Module):
    """Chain of NFBlocks. Cannot use nn.Sequential because each block returns (z, log_det)."""

    def __init__(self, C_z, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([NFBlock(C_z) for _ in range(num_blocks)])

    def forward(self, z):
        log_det = z.new_zeros(z.size(0))   # (B,)
        for block in self.blocks:
            z, ld = block(z)
            log_det = log_det + ld
        return z, log_det
