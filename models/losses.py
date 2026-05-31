"""Shared loss building blocks."""

import torch


class WeightedMSELoss(torch.nn.MSELoss):
    """MSE between (prediction * weights) and (target * weights)."""

    def forward(self, prediction, target, weights):
        return super().forward(prediction * weights, target * weights)
