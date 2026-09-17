import torch
from torch import nn
import torch.nn.functional as F

import config


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, target):
        valid_pixels = target != config.IGNORE_INDEX

        safe_target = target.clone()
        safe_target[~valid_pixels] = 0

        probabilities = torch.softmax(logits, dim=1)

        one_hot_target = F.one_hot(
            safe_target,
            num_classes=config.NUM_CLASSES,
        )

        one_hot_target = one_hot_target.permute(0, 3, 1, 2).float()

        valid_pixels = valid_pixels.unsqueeze(1)

        probabilities = probabilities * valid_pixels
        one_hot_target = one_hot_target * valid_pixels

        # Do not average background into the foreground loss.
        probabilities = probabilities[:, 1:]
        one_hot_target = one_hot_target[:, 1:]

        intersection = (
            probabilities * one_hot_target
        ).sum(dim=(0, 2, 3))

        denominator = (
            probabilities.sum(dim=(0, 2, 3))
            + one_hot_target.sum(dim=(0, 2, 3))
        )

        dice = (
            2.0 * intersection + self.epsilon
        ) / (
            denominator + self.epsilon
        )

        return 1.0 - dice.mean()


class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=config.IGNORE_INDEX
        )

        self.dice = DiceLoss()

    def forward(self, logits, target):
        ce = self.cross_entropy(logits, target)
        dice = self.dice(logits, target)

        return (
            config.CROSS_ENTROPY_WEIGHT * ce
            + config.DICE_LOSS_WEIGHT * dice
        )
