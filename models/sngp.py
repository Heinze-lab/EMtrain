"""
SNGP building blocks: a soft spectral-norm parametrization for Conv layers
and a Random-Feature Gaussian Process output head.

References:
    Liu, Lin, Padhy, Tran, Bedrax-Weiss, Lakshminarayanan.
    "Simple and Principled Uncertainty Estimation with Deterministic Deep
    Learning via Distance Awareness." NeurIPS 2020.

The implementations here are intentionally minimal and self-contained — they
do not depend on `edward2` or any external SNGP library.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


# -----------------------------------------------------------------------------
# Spectral normalization (soft / Bjorck-style upper-bound variant)
# -----------------------------------------------------------------------------

class _SoftSpectralNorm(nn.Module):
    """
    Parametrization that returns ``weight * coef / max(sigma_max(weight), coef)``.

    When the largest singular value of the weight matrix is below ``coef``, the
    weight is returned unchanged. When it exceeds ``coef``, the weight is
    rescaled so its spectral norm equals ``coef``. This is the formulation
    used in the SNGP paper (Liu et al. 2020), which tolerates a configurable
    Lipschitz upper bound rather than forcing every layer to ``sigma=1``.

    ``dim`` selects the output dimension of the weight tensor (0 for Conv,
    1 for ConvTranspose).
    """

    def __init__(self, weight, coef=0.95, n_power_iterations=1, dim=0, eps=1e-12):
        super().__init__()
        if coef <= 0:
            raise ValueError("coef must be positive")

        self.coef = float(coef)
        self.n_power_iterations = int(n_power_iterations)
        self.dim = int(dim)
        self.eps = float(eps)

        weight_mat = self._reshape_weight(weight)
        h, w = weight_mat.shape
        u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=self.eps)
        v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=self.eps)
        self.register_buffer("_u", u)
        self.register_buffer("_v", v)

    def _reshape_weight(self, weight):
        # Move the output dim to the front, then flatten the rest into one
        # column dim — same convention as torch's built-in spectral_norm.
        if self.dim != 0:
            weight = weight.transpose(0, self.dim)
        return weight.reshape(weight.shape[0], -1)

    def _power_iterate(self, weight_mat):
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                self._v.copy_(
                    F.normalize(weight_mat.t() @ self._u, dim=0, eps=self.eps)
                )
                self._u.copy_(
                    F.normalize(weight_mat @ self._v, dim=0, eps=self.eps)
                )

    def forward(self, weight):
        weight_mat = self._reshape_weight(weight)
        if self.training:
            self._power_iterate(weight_mat)
        sigma = self._u @ weight_mat @ self._v
        # Soft bound: rescale only when sigma > coef.
        factor = self.coef / torch.clamp(sigma, min=self.coef)
        return weight * factor


def apply_spectral_norm(module, coef=0.95, n_power_iterations=1):
    """
    Recursively wrap every ``Conv1/2/3d`` and ``ConvTranspose1/2/3d`` weight
    inside ``module`` with a soft spectral-norm parametrization.

    Returns the same module (mutated in place) for chaining.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            parametrize.register_parametrization(
                sub, "weight",
                _SoftSpectralNorm(sub.weight, coef=coef,
                                  n_power_iterations=n_power_iterations, dim=0)
            )
        elif isinstance(sub, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            parametrize.register_parametrization(
                sub, "weight",
                _SoftSpectralNorm(sub.weight, coef=coef,
                                  n_power_iterations=n_power_iterations, dim=1)
            )
    return module


# -----------------------------------------------------------------------------
# Random-Feature Gaussian Process head
# -----------------------------------------------------------------------------

class RandomFeatureGaussianProcess(nn.Module):
    """
    SNGP output head: random Fourier features approximation of an RBF-kernel
    Gaussian process, evaluated independently per spatial location.

    Input  : feature map of shape ``(B, C_in, *spatial)``.
    Outputs: dict with
        - ``logit``    : (B, num_classes, *spatial), the GP mean prediction
        - ``variance`` : (B, num_classes, *spatial), the predictive variance
                        (only computed when ``return_uncertainty=True``)

    During training (``self.training=True``) and when ``update_precision=True``,
    the precision matrix is updated EMA-style from each batch using the
    Laplace-approximation formula ``Σ ← m·Σ + (1-m)·Φᵀ diag(p(1-p)) Φ``.

    Call ``reset_precision()`` before the final epoch (or whenever you want
    a clean accumulation), and ``finalize_precision()`` once at the end of
    training to cache ``Σ⁻¹`` for inference.
    """

    def __init__(self,
                 in_features,
                 num_classes=1,
                 num_inducing=1024,
                 gp_kernel_scale=1.0,
                 gp_output_bias=0.0,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1.0,
                 normalize_input=True,
                 mean_field_factor=1.0):
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.num_inducing = int(num_inducing)
        self.gp_kernel_scale = float(gp_kernel_scale)
        self.gp_cov_momentum = float(gp_cov_momentum)
        self.gp_cov_ridge_penalty = float(gp_cov_ridge_penalty)
        self.mean_field_factor = float(mean_field_factor)

        # Random Fourier feature projection: phi(h) = sqrt(2/D) * cos(W h + b)
        rff_W = torch.randn(num_inducing, in_features) / self.gp_kernel_scale
        rff_b = torch.rand(num_inducing) * 2.0 * math.pi
        self.register_buffer("rff_W", rff_W)
        self.register_buffer("rff_b", rff_b)

        self.input_norm = nn.LayerNorm(in_features) if normalize_input else nn.Identity()

        # Trainable linear classifier (the GP "weights" beta).
        self.beta = nn.Linear(num_inducing, num_classes, bias=True)
        nn.init.constant_(self.beta.bias, gp_output_bias)

        # Precision matrix S = ridge*I + sum_t Φ_tᵀ diag(p_t(1-p_t)) Φ_t.
        # Stored as buffer so it lives in the state_dict and on the right device.
        precision = self.gp_cov_ridge_penalty * torch.eye(num_inducing)
        self.register_buffer("precision", precision)
        # Cached inverse for inference.
        self.register_buffer("precision_inv", torch.zeros_like(precision))
        # Whether the cached inverse is current.
        self.register_buffer("_inv_dirty", torch.tensor(True))

    # ------------------------------------------------------------------ utils
    def reset_precision(self):
        """Re-initialize the precision matrix to ``ridge * I``."""
        with torch.no_grad():
            self.precision.zero_()
            self.precision.add_(
                self.gp_cov_ridge_penalty
                * torch.eye(self.num_inducing, device=self.precision.device)
            )
            self._inv_dirty.fill_(True)

    def finalize_precision(self):
        """Cache ``precision_inv = precision^{-1}`` for fast inference."""
        with torch.no_grad():
            self.precision_inv.copy_(torch.linalg.inv(self.precision))
            self._inv_dirty.fill_(False)

    def _ensure_inv(self):
        if bool(self._inv_dirty.item()):
            self.finalize_precision()

    # ----------------------------------------------------------------- forward
    def _flatten_spatial(self, feat):
        B, C = feat.shape[0], feat.shape[1]
        spatial = feat.shape[2:]
        # (B, C, *spatial) -> (B, *spatial, C)
        perm = (0,) + tuple(range(2, feat.dim())) + (1,)
        return feat.permute(perm).reshape(-1, C), B, spatial

    def _unflatten_spatial(self, flat, B, spatial):
        # flat: (N, num_classes) -> (B, num_classes, *spatial)
        out = flat.reshape((B,) + tuple(spatial) + (self.num_classes,))
        # permute classes back to dim 1
        n_spatial = len(spatial)
        perm = (0, 1 + n_spatial) + tuple(range(1, 1 + n_spatial))
        return out.permute(perm).contiguous()

    def _compute_phi(self, h):
        h = self.input_norm(h)
        return math.sqrt(2.0 / self.num_inducing) * torch.cos(
            F.linear(h, self.rff_W, self.rff_b)
        )

    def update_precision(self, phi, logit):
        """EMA update of the Laplace precision matrix from a batch."""
        with torch.no_grad():
            prob = torch.sigmoid(logit)
            # p(1-p) per (sample, class). Average across classes so the
            # precision is shared. (Per-class precisions would multiply
            # storage by num_classes; rarely worth it.)
            w = (prob * (1.0 - prob)).mean(dim=-1)  # (N,)
            phi_w = phi * w.unsqueeze(-1)            # (N, D)
            update = phi.t() @ phi_w                  # (D, D)
            m = self.gp_cov_momentum
            self.precision.mul_(m).add_(update, alpha=(1.0 - m))
            self._inv_dirty.fill_(True)

    def forward(self, feat, return_uncertainty=False, update_precision=True):
        h, B, spatial = self._flatten_spatial(feat)
        phi = self._compute_phi(h)         # (N, D)
        logit = self.beta(phi)             # (N, num_classes)

        if self.training and update_precision:
            self.update_precision(phi.detach(), logit.detach())

        out = {"logit": self._unflatten_spatial(logit, B, spatial)}

        if return_uncertainty:
            self._ensure_inv()
            var = (phi @ self.precision_inv * phi).sum(dim=-1, keepdim=True)  # (N,1)
            if self.num_classes > 1:
                var = var.expand(-1, self.num_classes)
            out["variance"] = self._unflatten_spatial(var, B, spatial)

            # Mean-field-corrected probability (cf. Liu et al. eq. 13). Useful
            # if you want a calibrated confidence map rather than raw variance.
            scaled_logit = out["logit"] / torch.sqrt(
                1.0 + self.mean_field_factor * out["variance"]
            )
            out["mean_field_prob"] = torch.sigmoid(scaled_logit)

        return out
