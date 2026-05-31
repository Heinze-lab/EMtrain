"""
Original affinity + LSD model.
"""

import torch

from funlib.learn.torch.models import UNet, ConvPass

from .losses import WeightedMSELoss


class AffsLsdModel(torch.nn.Module):

    def __init__(self, num_fmaps):
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

    def forward(self, input):
        y = self.unet(input)
        affs = self.conv_affs(y)
        lsds = self.conv_lsds(y)
        return affs, lsds


class AffLsdLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.weighted_mse = WeightedMSELoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, loss_pred_affs, loss_affs, loss_affs_weights,
                loss_pred_lsds, loss_lsds):
        aff_loss = self.weighted_mse(loss_pred_affs, loss_affs, loss_affs_weights)
        lsd_loss = self.mse(loss_pred_lsds, loss_lsds)
        return aff_loss + lsd_loss


def build(model_config):
    """Factory entry point. See `emtrain.models.build_model`."""
    num_fmaps = model_config['num_fmaps']
    model = AffsLsdModel(num_fmaps=num_fmaps)
    loss = AffLsdLoss()
    output_keys = ['pred_affs', 'pred_lsds']
    return model, loss, output_keys
