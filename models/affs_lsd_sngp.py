"""
SNGP-augmented variant of AffsLsdModel.

Same U-Net trunk and (affs, lsds) heads as the original `affs_lsd` arch, plus:
  * soft spectral normalization wrapped around every Conv / ConvTranspose
    layer in the trunk (and optionally the heads), and
  * a Random-Feature Gaussian Process head that consumes a downsampled view
    of the trunk's penultimate features and emits a coarse-resolution
    uncertainty estimate.

Forward output (training):
    (pred_affs, pred_lsds, gp_logit)
Forward output (eval, when ``return_uncertainty`` is enabled):
    (pred_affs, pred_lsds, gp_logit, gp_uncertainty)

Switch to inference mode with ``model.enable_uncertainty()`` (which also
calls ``model.eval()`` and ensures the GP precision matrix has been
finalized).
"""

import torch.nn as nn
import torch.nn.functional as F

from funlib.learn.torch.models import UNet, ConvPass

from .losses import WeightedMSELoss
from .sngp import apply_spectral_norm, RandomFeatureGaussianProcess


class AffsLsdSngpModel(nn.Module):

    def __init__(self,
                 num_fmaps,
                 sn_coef=0.95,
                 sn_apply_to_heads=True,
                 gp_pool_kernel=(2, 2, 2),
                 gp_num_inducing=1024,
                 gp_num_classes=1,
                 gp_kernel_scale=1.0,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1.0,
                 gp_mean_field_factor=1.0):
        super().__init__()

        self.unet = UNet(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=5,
            downsample_factors=[
                [1, 2, 2],
                [1, 2, 2],
                [1, 2, 2]],
            kernel_size_down=[
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]]],
            kernel_size_up=[
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]],
                [[3, 3, 3], [3, 3, 3]]])

        self.conv_affs = ConvPass(num_fmaps, 3,  [[1, 1, 1]], activation='Sigmoid')
        self.conv_lsds = ConvPass(num_fmaps, 10, [[1, 1, 1]], activation='Sigmoid')

        # Spectral-norm wrapping. The trunk always gets it; the heads get it
        # by default (it's the version reported in the SNGP paper).
        apply_spectral_norm(self.unet, coef=sn_coef)
        if sn_apply_to_heads:
            apply_spectral_norm(self.conv_affs, coef=sn_coef)
            apply_spectral_norm(self.conv_lsds, coef=sn_coef)

        # Coarse-resolution pool on the trunk output.
        self.gp_pool_kernel = tuple(gp_pool_kernel) if gp_pool_kernel else None
        if self.gp_pool_kernel is not None:
            self.gp_pool = nn.AvgPool3d(self.gp_pool_kernel)
        else:
            self.gp_pool = nn.Identity()

        self.gp_head = RandomFeatureGaussianProcess(
            in_features=num_fmaps,
            num_classes=gp_num_classes,
            num_inducing=gp_num_inducing,
            gp_kernel_scale=gp_kernel_scale,
            gp_cov_momentum=gp_cov_momentum,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            mean_field_factor=gp_mean_field_factor,
        )

        self._return_uncertainty = False

    # ------------------------------------------------------------------ utils
    def enable_uncertainty(self):
        """Switch to inference mode and emit variance from forward()."""
        self.eval()
        self._return_uncertainty = True
        # Cache the precision inverse now so the first forward pass isn't slow.
        self.gp_head.finalize_precision()

    def disable_uncertainty(self):
        self._return_uncertainty = False

    def reset_gp_precision(self):
        """Reset the GP precision matrix. Call before the final training
        phase if you only want it accumulated over recent iterations."""
        self.gp_head.reset_precision()

    def finalize_gp_precision(self):
        """Cache the inverse precision. Call once at end of training."""
        self.gp_head.finalize_precision()

    # ---------------------------------------------------------------- forward
    def forward(self, input):
        y = self.unet(input)
        affs = self.conv_affs(y)
        lsds = self.conv_lsds(y)

        h = self.gp_pool(y)
        gp_out = self.gp_head(
            h,
            return_uncertainty=self._return_uncertainty,
            update_precision=self.training,
        )

        if self._return_uncertainty:
            return affs, lsds, gp_out["logit"], gp_out["variance"]
        return affs, lsds, gp_out["logit"]


class AffLsdSngpLoss(nn.Module):
    """
    Original aff+lsd loss plus a per-coarse-voxel BCE on the GP logit.

    The coarse target is computed inside the loss by mean-pooling the
    ground-truth affinities to the GP head's spatial resolution. If
    ``gp_num_classes == 1``, the 3 affinity channels are first averaged.
    """

    def __init__(self,
                 gp_pool_kernel=(2, 2, 2),
                 gp_loss_weight=0.5,
                 gp_num_classes=1):
        super().__init__()
        self.weighted_mse = WeightedMSELoss()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.gp_pool_kernel = tuple(gp_pool_kernel)
        self.gp_loss_weight = float(gp_loss_weight)
        self.gp_num_classes = int(gp_num_classes)

    def _coarse_target(self, affs):
        if self.gp_num_classes == 1:
            target = affs.mean(dim=1, keepdim=True)
        else:
            assert affs.shape[1] == self.gp_num_classes, (
                f"GP num_classes={self.gp_num_classes} but affs has "
                f"{affs.shape[1]} channels"
            )
            target = affs
        return F.avg_pool3d(target, self.gp_pool_kernel)

    def forward(self, loss_pred_affs, loss_affs, loss_affs_weights,
                loss_pred_lsds, loss_lsds, loss_gp_logit):
        aff_loss = self.weighted_mse(loss_pred_affs, loss_affs, loss_affs_weights)
        lsd_loss = self.mse(loss_pred_lsds, loss_lsds)
        coarse_target = self._coarse_target(loss_affs)
        gp_loss = self.bce(loss_gp_logit, coarse_target)
        return aff_loss + lsd_loss + self.gp_loss_weight * gp_loss


def build(model_config):
    """Factory entry point. See `emtrain.models.build_model`."""
    num_fmaps = model_config['num_fmaps']

    sngp_cfg = model_config.get('sngp', {})
    sn_coef                = sngp_cfg.get('sn_coef', 0.95)
    sn_apply_to_heads      = sngp_cfg.get('sn_apply_to_heads', True)
    gp_pool_kernel         = sngp_cfg.get('gp_pool_kernel', [2, 2, 2])
    gp_num_inducing        = sngp_cfg.get('gp_num_inducing', 1024)
    gp_num_classes         = sngp_cfg.get('gp_num_classes', 1)
    gp_kernel_scale        = sngp_cfg.get('gp_kernel_scale', 1.0)
    gp_cov_momentum        = sngp_cfg.get('gp_cov_momentum', 0.999)
    gp_cov_ridge_penalty   = sngp_cfg.get('gp_cov_ridge_penalty', 1.0)
    gp_mean_field_factor   = sngp_cfg.get('gp_mean_field_factor', 1.0)
    gp_loss_weight         = sngp_cfg.get('gp_loss_weight', 0.5)

    model = AffsLsdSngpModel(
        num_fmaps=num_fmaps,
        sn_coef=sn_coef,
        sn_apply_to_heads=sn_apply_to_heads,
        gp_pool_kernel=gp_pool_kernel,
        gp_num_inducing=gp_num_inducing,
        gp_num_classes=gp_num_classes,
        gp_kernel_scale=gp_kernel_scale,
        gp_cov_momentum=gp_cov_momentum,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        gp_mean_field_factor=gp_mean_field_factor,
    )
    loss = AffLsdSngpLoss(
        gp_pool_kernel=gp_pool_kernel,
        gp_loss_weight=gp_loss_weight,
        gp_num_classes=gp_num_classes,
    )
    output_keys = ['pred_affs', 'pred_lsds', 'gp_logit']
    return model, loss, output_keys
