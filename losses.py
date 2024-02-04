import torch


# Dice Loss Function
def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()
