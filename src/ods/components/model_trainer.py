"""
Model Trainer
=============
Handles:
  • Training loop with AMP (mixed precision)
  • Gradient clipping
  • Optimizer & scheduler construction (SGD, Adam, AdamW + warmup)
  • TensorBoard logging
  • MLflow run management
  • Checkpoint saving (best + periodic)
  • Resume from checkpoint
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.ods.entity.config_entity import ModelConfig, TrainingConfig
from src.ods.evaluation.metrics import CombinedEvaluator
from src.ods.losses.losses import compute_total_detection_loss, get_seg_loss
from src.ods.models.model import ODSModel
from src.ods.utils.common import get_device, save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    Full training + validation loop.

    Instantiate once per experiment:
        trainer = Trainer(model, train_cfg, model_cfg)
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model: ODSModel,
        train_cfg: TrainingConfig,
        model_cfg: ModelConfig,
    ):
        self.model = model
        self.cfg = train_cfg
        self.model_cfg = model_cfg
        self.device = get_device()
        self.model.to(self.device)

        # Segmentation loss (detection loss comes from FasterRCNN itself)
        self.seg_loss_fn = None
        if train_cfg.task in ("segmentation", "both"):
            self.seg_loss_fn = get_seg_loss(
                train_cfg.segmentation_loss,
                num_classes=model_cfg.num_classes_seg,
                ignore_index=255,
            )

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Scheduler (built after optimizer; OneCycle needs train_loader length)
        self.scheduler = None
        self.scheduler_name = train_cfg.scheduler

        # AMP scaler
        self.scaler = GradScaler(enabled=train_cfg.use_amp and self.device.type == "cuda")

        # Logging
        self.writer = SummaryWriter(log_dir=str(train_cfg.logs_dir))
        train_cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        train_cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.start_epoch = 0
        self.best_metric = 0.0
        self.global_step = 0

        # Resume
        if train_cfg.resume_checkpoint:
            self._resume(train_cfg.resume_checkpoint)

        # Evaluator
        self.evaluator = CombinedEvaluator(
            task=train_cfg.task,
            num_classes_seg=model_cfg.num_classes_seg,
            num_classes_det=model_cfg.num_classes_det,
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def _build_optimizer(self):
        # Separate backbone and head params for differential LR
        backbone_params = list(self.model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.model.parameters()
                       if id(p) not in backbone_ids and p.requires_grad]
        backbone_trainable = [p for p in backbone_params if p.requires_grad]

        param_groups = [
            {"params": backbone_trainable, "lr": self.cfg.lr * 0.1},   # lower LR for backbone
            {"params": head_params, "lr": self.cfg.lr},
        ]

        opt_name = self.cfg.optimizer.lower()
        if opt_name == "sgd":
            return SGD(param_groups, momentum=self.cfg.momentum,
                       weight_decay=self.cfg.weight_decay)
        elif opt_name == "adam":
            return Adam(param_groups, weight_decay=self.cfg.weight_decay)
        elif opt_name == "adamw":
            return AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, steps_per_epoch: int):
        name = self.scheduler_name.lower()
        total_epochs = self.cfg.epochs

        if name == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=total_epochs - self.cfg.warmup_epochs, eta_min=1e-6
            )
        elif name == "step":
            return StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif name == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=[self.cfg.lr * 0.1, self.cfg.lr],
                steps_per_epoch=steps_per_epoch,
                epochs=total_epochs,
                pct_start=0.1,
            )
        elif name == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="max", patience=5, factor=0.5)
        else:
            raise ValueError(f"Unknown scheduler: {name}")

    # ── Warmup LR ─────────────────────────────────────────────────────────────

    def _warmup_lr(self, epoch: int) -> None:
        if epoch < self.cfg.warmup_epochs:
            warmup_factor = (epoch + 1) / self.cfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * warmup_factor

    # ── Main fit loop ─────────────────────────────────────────────────────────

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        # Build scheduler now that we have loader length
        self.scheduler = self._build_scheduler(len(train_loader))

        # ── MLflow run ────────────────────────────────────────────────────────
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.experiment_name)

        with mlflow.start_run(run_name=self.cfg.experiment_name) as run:
            # Log params
            mlflow.log_params({
                "task": self.cfg.task,
                "backbone": self.model_cfg.backbone,
                "optimizer": self.cfg.optimizer,
                "lr": self.cfg.lr,
                "scheduler": self.cfg.scheduler,
                "epochs": self.cfg.epochs,
                "det_loss": self.cfg.detection_loss,
                "seg_loss": self.cfg.segmentation_loss,
                "description": self.cfg.description,
            })
            logger.info(f"MLflow run ID: {run.info.run_id}")

            for epoch in range(self.start_epoch, self.cfg.epochs):
                self._warmup_lr(epoch)
                current_lr = self.optimizer.param_groups[-1]["lr"]

                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch [{epoch+1}/{self.cfg.epochs}]  LR={current_lr:.6f}")

                # Train
                train_metrics = self._train_epoch(train_loader, epoch)

                # Validate
                if (epoch + 1) % self.cfg.eval_every_n_epochs if hasattr(self.cfg, "eval_every_n_epochs") else True:
                    val_metrics = self._val_epoch(val_loader, epoch)
                else:
                    val_metrics = {}

                # Scheduler step
                if self.scheduler_name == "plateau":
                    self.scheduler.step(val_metrics.get("mIoU", val_metrics.get("mAP@0.5", 0)))
                elif self.scheduler_name != "onecycle":
                    self.scheduler.step()

                # TensorBoard
                self._log_tensorboard(train_metrics, val_metrics, epoch)

                # MLflow
                mlflow.log_metrics({**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}, step=epoch)

                # Checkpoint
                primary_metric = val_metrics.get("mIoU", val_metrics.get("mAP@0.5", 0))
                is_best = primary_metric > self.best_metric
                if is_best:
                    self.best_metric = primary_metric

                self.model.save_weights(
                    self.cfg.checkpoints_dir / f"epoch_{epoch+1}.pth",
                    extra={"epoch": epoch + 1, "metrics": val_metrics},
                )
                if is_best:
                    self.model.save_weights(
                        self.cfg.checkpoints_dir / "best_model.pth",
                        extra={"epoch": epoch + 1, "metrics": val_metrics},
                    )
                    mlflow.log_artifact(
                        str(self.cfg.checkpoints_dir / "best_model.pth")
                    )
                    logger.info(f"  ✓ New best: {primary_metric:.4f}")

            self.writer.close()
            logger.info("Training complete.")

    # ── Training epoch ────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_det_loss = 0.0
        total_seg_loss = 0.0
        n = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["images"].to(self.device)
            seg_masks = batch["seg_masks"].to(self.device)
            boxes = [b.to(self.device) for b in batch["boxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            self.optimizer.zero_grad()

            with autocast(enabled=self.cfg.use_amp and self.device.type == "cuda"):
                loss = torch.tensor(0.0, device=self.device)

                # Detection
                if self.cfg.task in ("detection", "both"):
                    targets = self.model.det_head.prepare_targets(boxes, labels, self.device)
                    self.model.det_head.train()
                    det_losses = self.model.det_head(images, targets)
                    det_loss = compute_total_detection_loss(det_losses)
                    loss = loss + det_loss
                    total_det_loss += det_loss.item()

                # Segmentation
                if self.cfg.task in ("segmentation", "both"):
                    from src.ods.models.backbone import build_backbone
                    features = self.model.backbone(images)
                    seg_logits = self.model.seg_head(features, (images.shape[2], images.shape[3]))
                    seg_loss = self.seg_loss_fn(seg_logits, seg_masks)
                    loss = loss + seg_loss
                    total_seg_loss += seg_loss.item()

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_name == "onecycle":
                self.scheduler.step()

            total_loss += loss.item()
            n += 1
            self.global_step += 1

            if batch_idx % 50 == 0:
                logger.info(
                    f"  [{batch_idx}/{len(loader)}]  "
                    f"loss={loss.item():.4f}  "
                    f"det={total_det_loss/(n+1e-8):.4f}  "
                    f"seg={total_seg_loss/(n+1e-8):.4f}"
                )

        return {
            "train_loss": total_loss / max(n, 1),
            "train_det_loss": total_det_loss / max(n, 1),
            "train_seg_loss": total_seg_loss / max(n, 1),
            "lr": self.optimizer.param_groups[-1]["lr"],
        }

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.evaluator.reset()

        for batch in loader:
            images = batch["images"].to(self.device)
            seg_masks = batch["seg_masks"].to(self.device)
            boxes = [b.to(self.device) for b in batch["boxes"]]
            labels = [l.to(self.device) for l in batch["labels"]]

            # Detection predictions
            if self.cfg.task in ("detection", "both"):
                self.model.det_head.eval()
                det_preds = self.model.det_head(images)
                # Format for evaluator
                gt_targets = self.model.det_head.prepare_targets(boxes, labels, self.device)
                self.evaluator.update_detection(det_preds, gt_targets)

            # Segmentation predictions
            if self.cfg.task in ("segmentation", "both"):
                features = self.model.backbone(images)
                seg_logits = self.model.seg_head(features, (images.shape[2], images.shape[3]))
                seg_preds = seg_logits.argmax(dim=1)  # [B, H, W]
                self.evaluator.update_segmentation(seg_preds, seg_masks)

        return self.evaluator.compute()

    # ── TensorBoard ───────────────────────────────────────────────────────────

    def _log_tensorboard(
        self,
        train_metrics: Dict,
        val_metrics: Dict,
        epoch: int,
    ) -> None:
        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        self.writer.flush()

    # ── Resume ────────────────────────────────────────────────────────────────

    def _resume(self, checkpoint_path: str) -> None:
        logger.info(f"Resuming from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.best_metric = ckpt.get("best_metric", 0.0)
        logger.info(f"Resumed from epoch {self.start_epoch}")