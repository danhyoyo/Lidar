import torch.nn.functional as F


def l1_loss(pred, target, mask):
    mask = mask.unsqueeze(1).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss

def smooth_l1_loss(pred, target, mask):
    mask = mask.unsqueeze(1).expand_as(pred).float()
    loss = F.smooth_l1_loss(pred * mask, target * mask, reduction="sum")
    loss = loss / (mask.sum() + 1e-4)
    return loss
