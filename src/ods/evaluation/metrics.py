"""
Evaluation Metrics
==================
Detection  : mAP @ IoU 0.5 and 0.5:0.95  (torchmetrics)
Segmentation: mIoU, per-class IoU, pixel accuracy  (numpy-based, fast)

We avoid heavy libraries like pycocotools for simplicity on laptops.
torchmetrics handles the COCO-style mAP correctly.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  DETECTION — mAP
# ═════════════════════════════════════════════════════════════════════

class DetectionEvaluator:
    """
    Accumulates predictions and targets across batches,
    then computes mAP using torchmetrics.

    Usage:
        ev = DetectionEvaluator(iou_thresholds=[0.5, 0.75])
        for batch in val_loader:
            preds = model.forward(batch["images"])
            ev.update(preds, targets)
        results = ev.compute()
        ev.reset()
    """

    def __init__(self, iou_thresholds: List[float] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.75]
        self.metric = MeanAveragePrecision(
            iou_thresholds=self.iou_thresholds,
            class_metrics=True,
        )

    def update(
        self,
        preds: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        preds   : list of dicts per image, keys: boxes, scores, labels
        targets : list of dicts per image, keys: boxes, labels
        """
        self.metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        results = self.metric.compute()
        output = {
            "mAP@0.5": float(results.get("map_50", 0)),
            "mAP@0.75": float(results.get("map_75", 0)),
            "mAP@0.5:0.95": float(results.get("map", 0)),
        }
        logger.info(
            f"Detection  mAP@0.5={output['mAP@0.5']:.4f}  "
            f"mAP@0.5:0.95={output['mAP@0.5:0.95']:.4f}"
        )
        return output

    def reset(self) -> None:
        self.metric.reset()


# ═════════════════════════════════════════════════════════════════════
#  SEGMENTATION — mIoU
# ═════════════════════════════════════════════════════════════════════

class SegmentationEvaluator:
    """
    Confusion-matrix-based mIoU evaluator.
    Faster than per-image IoU computation.

    IoU_c = TP_c / (TP_c + FP_c + FN_c)
          = conf[c,c] / (row_sum + col_sum - conf[c,c])
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(
        self,
        preds: torch.Tensor,    # [B, H, W] argmax predictions
        targets: torch.Tensor,  # [B, H, W] ground truth (0..num_classes-1 or 255)
    ) -> None:
        preds_np = preds.cpu().numpy().ravel()
        targets_np = targets.cpu().numpy().ravel()

        # Remove ignore pixels
        valid = targets_np != self.ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Accumulate confusion matrix using bincount trick (fast)
        mask = (targets_np >= 0) & (targets_np < self.num_classes) & \
               (preds_np >= 0) & (preds_np < self.num_classes)
        indices = self.num_classes * targets_np[mask] + preds_np[mask]
        conf = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion += conf.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        conf = self.confusion.astype(np.float64)
        tp = np.diag(conf)
        fp = conf.sum(axis=0) - tp
        fn = conf.sum(axis=1) - tp

        iou_per_class = tp / (tp + fp + fn + 1e-8)
        # Only average over classes that appear in ground truth
        valid_classes = conf.sum(axis=1) > 0
        miou = iou_per_class[valid_classes].mean()

        pixel_acc = tp.sum() / (conf.sum() + 1e-8)

        results = {
            "mIoU": float(miou),
            "pixel_accuracy": float(pixel_acc),
        }
        # Per-class IoU
        from src.ods.constants import CITYSCAPES_CLASSES
        for i, cls_name in enumerate(CITYSCAPES_CLASSES):
            results[f"IoU_{cls_name}"] = float(iou_per_class[i])

        logger.info(
            f"Segmentation  mIoU={miou:.4f}  pixel_acc={pixel_acc:.4f}"
        )
        return results

    def reset(self) -> None:
        self.confusion[:] = 0

    @property
    def miou(self) -> float:
        return self.compute()["mIoU"]


# ═════════════════════════════════════════════════════════════════════
#  COMBINED EVALUATOR
# ═════════════════════════════════════════════════════════════════════

class CombinedEvaluator:
    """Wrapper that holds both evaluators. Use for 'both' task mode."""

    def __init__(
        self,
        task: str,
        num_classes_seg: int = 19,
        num_classes_det: int = 8,
        ignore_index: int = 255,
        iou_thresholds: List[float] = None,
    ):
        self.task = task
        self.det_ev = (
            DetectionEvaluator(iou_thresholds)
            if task in ("detection", "both") else None
        )
        self.seg_ev = (
            SegmentationEvaluator(num_classes_seg, ignore_index)
            if task in ("segmentation", "both") else None
        )

    def update_detection(self, preds, targets):
        if self.det_ev:
            self.det_ev.update(preds, targets)

    def update_segmentation(self, preds, targets):
        if self.seg_ev:
            self.seg_ev.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        results = {}
        if self.det_ev:
            results.update(self.det_ev.compute())
        if self.seg_ev:
            results.update(self.seg_ev.compute())
        return results

    def reset(self):
        if self.det_ev:
            self.det_ev.reset()
        if self.seg_ev:
            self.seg_ev.reset()