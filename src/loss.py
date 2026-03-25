import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    num = 2 * (probs * targets).sum(dim=(2, 3))
    den = (probs + targets).sum(dim=(2, 3)) + eps
    return (1 - num / den).mean()


def bce_dice_loss(logits, targets):
    return (
        F.binary_cross_entropy_with_logits(logits, targets)
        + dice_loss(logits, targets)
    )
