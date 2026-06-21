import math
import torch
import torch.nn.functional as F


def soft_clamp(x, n=5.0):
    """Differentiable clamp to [-n, n]. Keeps mu/log_sigma bounded without zero gradients."""
    return n * torch.tanh(x / n)


class Normal:
    """Diagonal Gaussian parameterized by (mu, log_sigma) with soft clamping."""

    def __init__(self, mu, log_sigma):
        self.mu = soft_clamp(mu)
        self.log_sigma = soft_clamp(log_sigma)

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps, eps

    def log_prob(self, z):
        return -0.5 * (math.log(2 * math.pi) + 2 * self.log_sigma + ((z - self.mu) / self.sigma) ** 2)

    def kl(self, p):
        """Analytic KL(self || p). Both must be Normal. Returns per-element values."""
        return (p.log_sigma - self.log_sigma
                + (self.sigma ** 2 + (self.mu - p.mu) ** 2) / (2 * p.sigma ** 2)
                - 0.5)

    def scale_temperature(self, t):
        """Return a new Normal with sigma scaled by t (for generation temperature)."""
        return Normal(self.mu, self.log_sigma + math.log(t))


class DiscMixLogistic:
    """Mixture of K discretized logistic distributions for 3-channel (RGB) images.

    Logits layout (dim 1): [K mix_weights | K*9 per-channel params]
    Per-channel params (for K mixtures, 3 channels): means(3) + log_scales(3) + coeffs(3)
    Channel dependency: green = mu_g + c0*r, blue = mu_b + c1*r + c2*g.
    Input x is expected in [-1, 1].
    """

    def __init__(self, logits, num_bits=8):
        B, C, H, W = logits.shape
        self.K = C // 10
        self.bins = 2 ** num_bits

        self.mix_logits = logits[:, :self.K]  # (B, K, H, W)
        params = logits[:, self.K:].reshape(B, self.K, 9, H, W)
        self.means = params[:, :, :3]                         # (B, K, 3, H, W)
        self.log_scales = soft_clamp(params[:, :, 3:6], n=7.) # (B, K, 3, H, W)
        self.coeffs = torch.tanh(params[:, :, 6:])            # (B, K, 3, H, W)

    def log_prob(self, x):
        """Log-likelihood of x. x: (B, 3, H, W) in [-1, 1]. Returns (B, H, W)."""
        B, _, H, W = x.shape
        x = x.unsqueeze(1).expand(-1, self.K, -1, -1, -1)  # (B, K, 3, H, W)

        # Channel-dependent means
        m0 = self.means[:, :, 0]
        m1 = self.means[:, :, 1] + self.coeffs[:, :, 0] * x[:, :, 0]
        m2 = (self.means[:, :, 2]
              + self.coeffs[:, :, 1] * x[:, :, 0]
              + self.coeffs[:, :, 2] * x[:, :, 1])
        means = torch.stack([m0, m1, m2], dim=2)  # (B, K, 3, H, W)

        inv_scale = torch.exp(-self.log_scales)
        centered = (x - means) * inv_scale  # (B, K, 3, H, W)

        half_bin = 1.0 / (self.bins - 1)   # half a pixel step in [-1, 1] space

        cdf_plus = torch.sigmoid(centered + half_bin * inv_scale)
        cdf_minus = torch.sigmoid(centered - half_bin * inv_scale)
        cdf_delta = cdf_plus - cdf_minus

        # Numerically stable fallback: when scale is very small the CDF difference
        # underflows; use log(PDF * bin_width) = logsigmoid(c) + logsigmoid(-c) + log(2*half_bin/scale)
        log_mid = torch.where(
            cdf_delta > 1e-5,
            torch.log(cdf_delta.clamp(min=1e-10)),
            F.logsigmoid(centered) + F.logsigmoid(-centered)
            + math.log(2 * half_bin) - self.log_scales,
        )
        log_low = torch.log(cdf_plus.clamp(min=1e-10))           # at x = -1
        log_high = torch.log((1.0 - cdf_minus).clamp(min=1e-10)) # at x = +1

        log_prob = torch.where(x < -0.999, log_low,
                               torch.where(x > 0.99, log_high, log_mid))
        log_prob = log_prob.sum(dim=2)  # sum over RGB → (B, K, H, W)

        log_mix = F.log_softmax(self.mix_logits, dim=1) + log_prob
        return torch.logsumexp(log_mix, dim=1)  # (B, H, W)

    def sample(self, t=1.0):
        """Sample an image. t < 1 gives sharper outputs. Returns (B, 3, H, W) in [-1, 1]."""
        B, K, H, W = self.mix_logits.shape

        # Gumbel-max trick to select mixture component
        u = torch.rand_like(self.mix_logits).clamp(1e-5, 1 - 1e-5)
        gumbel = -torch.log(-torch.log(u))
        idx = (self.mix_logits / t + gumbel).argmax(dim=1)  # (B, H, W)
        one_hot = F.one_hot(idx, K).permute(0, 3, 1, 2).float()  # (B, K, H, W)

        means = (self.means * one_hot.unsqueeze(2)).sum(1)       # (B, 3, H, W)
        log_sc = (self.log_scales * one_hot.unsqueeze(2)).sum(1)
        coeffs = (self.coeffs * one_hot.unsqueeze(2)).sum(1)

        # Inverse logistic CDF: log(u/(1-u))
        u = torch.rand_like(means).clamp(1e-5, 1 - 1e-5)
        sample = means + log_sc.exp() * t * (torch.log(u) - torch.log1p(-u))

        r = sample[:, 0].clamp(-1, 1)
        g = (sample[:, 1] + coeffs[:, 0] * r).clamp(-1, 1)
        b = (sample[:, 2] + coeffs[:, 1] * r + coeffs[:, 2] * g).clamp(-1, 1)
        return torch.stack([r, g, b], dim=1)
