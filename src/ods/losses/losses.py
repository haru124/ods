"""
Loss Functions  —  Switchable via config
=========================================

Detection losses (returned by FasterRCNN's own forward, but we can swap
the classification loss component):
  • "focal"    : Focal Loss for classification (handles class imbalance)
  • "smoothl1" : standard regression loss (default)
  • "giou"     : Generalised IoU loss for box regression

Segmentation losses:
  • "cross_entropy" : standard per-pixel CE (baseline)
  • "dice"          : Dice Loss (handles class imbalance via overlap)
  • "focal"         : Focal Loss per pixel
  • "combo"         : 0.5 * CE + 0.5 * Dice  (often best)

Teaching note — when to use which:
  CE  : Simple, fast. Works when class distribution is balanced.
  Focal: Downweights easy pixels/boxes → useful when background dominates.
  Dice: Optimises overlap directly → good for small objects/classes.
  GIoU: Penalises non-overlapping boxes → often better convergence than L1.
  Combo: Hedges between reconstruction (Dice) and calibration (CE).
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  SEGMENTATION LOSSES
# ═════════════════════════════════════════════════════════════════════

class CrossEntropySegLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    We compute per-class Dice and average over valid classes.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255,
                 smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)            # [B, C, H, W]
        B, C, H, W = probs.shape

        # One-hot encode targets, masking out ignore regions
        valid_mask = (targets != self.ignore_index)  # [B, H, W]
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0

        one_hot = F.one_hot(targets_clamped, num_classes=C)  # [B, H, W, C]
        one_hot = one_hot.permute(0, 3, 1, 2).float()        # [B, C, H, W]

        # Apply valid mask per class
        mask = valid_mask.unsqueeze(1).float()
        probs = probs * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
        dice_per_class = 1 - (2 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_per_class.mean()


class FocalSegLoss(nn.Module):
    """
    Focal Loss for segmentation.
    FL(p) = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 ignore_index: int = 255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        # Flatten
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        targets_flat = targets.reshape(-1)                          # [B*H*W]

        # Remove ignore pixels
        valid = targets_flat != self.ignore_index
        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]

        if targets_flat.numel() == 0:
            return logits_flat.sum() * 0.0

        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        probs = torch.exp(-ce)
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        return (focal_weight * ce).mean()


class ComboLoss(nn.Module):
    """0.5 * CrossEntropy + 0.5 * Dice  (balance between both)."""
    def __init__(self, num_classes: int, ignore_index: int = 255,
                 ce_weight: float = 0.5):
        super().__init__()
        self.ce = CrossEntropySegLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(logits, targets) + \
               (1 - self.ce_weight) * self.dice(logits, targets)


# ═════════════════════════════════════════════════════════════════════
#  DETECTION LOSSES (supplement to FasterRCNN's built-in losses)
# ═════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multi-class classification.
    Plugged into the RPN classification in place of standard CE.
    Note: FasterRCNN already returns a loss dict — we use this
    to post-process the returned classification loss if needed.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class GIoULoss(nn.Module):
    """
    Generalised Intersection over Union loss.
    Better than L1/SmoothL1 for non-overlapping boxes.
    Uses torchvision's built-in GIoU implementation.
    """
    def forward(
        self,
        pred_boxes: torch.Tensor,   # [N, 4] XYXY
        target_boxes: torch.Tensor, # [N, 4] XYXY
    ) -> torch.Tensor:
        from torchvision.ops import generalized_box_iou
        if pred_boxes.numel() == 0:
            return pred_boxes.sum() * 0
        giou = generalized_box_iou(pred_boxes, target_boxes)
        # GIoU ∈ [-1, 1]; loss = 1 - GIoU (want to maximise GIoU)
        diag = torch.diag(giou)
        return (1 - diag).mean()


# ═════════════════════════════════════════════════════════════════════
#  FACTORIES
# ═════════════════════════════════════════════════════════════════════

def get_seg_loss(name: str, num_classes: int, ignore_index: int = 255) -> nn.Module:
    """
    Factory for segmentation loss.
    Called once at training setup; the returned module is used each step.
    """
    name = name.lower()
    if name == "cross_entropy":
        loss = CrossEntropySegLoss(ignore_index=ignore_index)
    elif name == "dice":
        loss = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
    elif name == "focal":
        loss = FocalSegLoss(ignore_index=ignore_index)
    elif name == "combo":
        loss = ComboLoss(num_classes=num_classes, ignore_index=ignore_index)
    else:
        raise ValueError(
            f"Unknown seg loss: '{name}'. "
            "Choose: cross_entropy | dice | focal | combo"
        )
    logger.info(f"Segmentation loss: {name}")
    return loss


def get_det_aux_loss(name: str) -> Optional[nn.Module]:
    """
    Return an *auxiliary* detection loss module.
    FasterRCNN's built-in losses are always computed.
    This is returned as an extra loss to add if desired.
    Currently: focal and giou auxiliary losses.
    """
    name = name.lower()
    if name in ("focal", "focal_loss"):
        return FocalLoss()
    elif name in ("giou", "giou_loss"):
        return GIoULoss()
    else:
        return None   # FasterRCNN default losses only


def compute_total_detection_loss(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    FasterRCNN.forward() returns a dict of losses.
    We sum them to get the total scalar loss for backprop.
    Logs individual components for TensorBoard.
    """
    total = sum(loss_dict.values())
    return total