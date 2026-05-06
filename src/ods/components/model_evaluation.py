"""
Model Evaluation Component
==========================
Runs full evaluation on val or test split, logs metrics to MLflow,
produces per-class IoU table, and writes a JSON report to disk.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import mlflow
import torch
from torch.utils.data import DataLoader

from src.ods.entity.config_entity import EvaluationConfig, ModelConfig, TrainingConfig
from src.ods.evaluation.metrics import CombinedEvaluator
from src.ods.models.model import ODSModel
from src.ods.utils.common import get_device, save_json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(
        self,
        model: ODSModel,
        eval_cfg: EvaluationConfig,
        train_cfg: TrainingConfig,
    ):
        self.model = model
        self.cfg = eval_cfg
        self.train_cfg = train_cfg
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()

        self.evaluator = CombinedEvaluator(
            task=train_cfg.task,
            num_classes_seg=eval_cfg.num_classes_seg,
            num_classes_det=eval_cfg.num_classes_det,
            ignore_index=eval_cfg.ignore_index,
            iou_thresholds=eval_cfg.det_iou_thresholds,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str = "val") -> Dict[str, float]:
        self.evaluator.reset()

        for batch in loader:
            images = batch["images"].to(self.device)
            seg_masks = batch["seg_masks"].to(self.device)
            boxes = [b.to(self.device) for b in batch["boxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            if self.train_cfg.task in ("detection", "both"):
                self.model.det_head.eval()
                det_preds = self.model.det_head(images)
                gt_targets = self.model.det_head.prepare_targets(boxes, labels, self.device)
                self.evaluator.update_detection(det_preds, gt_targets)

            if self.train_cfg.task in ("segmentation", "both"):
                features = self.model.backbone(images)
                seg_logits = self.model.seg_head(
                    features, (images.shape[2], images.shape[3])
                )
                seg_preds = seg_logits.argmax(dim=1)
                self.evaluator.update_segmentation(seg_preds, seg_masks)

        results = self.evaluator.compute()
        self._log_results(results, split)
        return results

    def _log_results(self, results: Dict, split: str) -> None:
        logger.info(f"\nEvaluation Results [{split}]:")
        for k, v in results.items():
            if not k.startswith("IoU_"):
                logger.info(f"  {k:<25s}: {v:.4f}")

        # Per-class IoU table
        per_class = {k: v for k, v in results.items() if k.startswith("IoU_")}
        if per_class:
            logger.info("\n  Per-class IoU:")
            for k, v in sorted(per_class.items(), key=lambda x: -x[1]):
                bar = "█" * int(v * 20)
                logger.info(f"    {k[4:]:20s} {v:.3f}  {bar}")

        # Save report
        report_path = (
            Path(self.train_cfg.checkpoints_dir).parent
            / f"eval_{split}_results.json"
        )
        save_json(results, report_path)
        logger.info(f"Report saved: {report_path}")

        # MLflow
        try:
            mlflow.set_tracking_uri(self.train_cfg.mlflow_tracking_uri)
            with mlflow.start_run(
                run_name=f"{self.train_cfg.experiment_name}_eval_{split}",
                nested=True,
            ):
                mlflow.log_metrics({f"{split}_{k}": v for k, v in results.items()
                                    if not k.startswith("IoU_")})
        except Exception as e:
            logger.warning(f"MLflow logging skipped: {e}")