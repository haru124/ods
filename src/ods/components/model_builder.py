"""
Model Builder Component
=======================
Builds the ODSModel from config, optionally loads pretrained/partial weights,
logs parameter counts, and returns the model ready for training.
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from src.ods.entity.config_entity import ModelConfig, TrainingConfig
from src.ods.models.model import ODSModel
from src.ods.utils.common import count_parameters, get_device

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainingConfig):
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

    def build(self, weight_path: Optional[str] = None) -> ODSModel:
        """
        Build model for the configured task.
        If weight_path is given, loads weights with strict=False (allows
        partial loads — e.g. backbone-only checkpoint into full model).
        """
        model = ODSModel(self.model_cfg, task=self.train_cfg.task)
        param_info = count_parameters(model)

        if weight_path and Path(weight_path).exists():
            model.load_weights(weight_path, strict=False)
            logger.info(f"Loaded weights from: {weight_path}")
        elif self.train_cfg.resume_checkpoint:
            model.load_weights(self.train_cfg.resume_checkpoint, strict=False)
            logger.info(f"Resumed from: {self.train_cfg.resume_checkpoint}")

        self._log_model_summary(model, param_info)
        return model

    def _log_model_summary(self, model: ODSModel, param_info: dict) -> None:
        device = get_device()
        logger.info(
            f"\nModel Summary\n"
            f"  Task      : {model.task}\n"
            f"  Backbone  : {self.model_cfg.backbone}\n"
            f"  Decoder   : {self.model_cfg.decoder}\n"
            f"  Params    : {param_info['total']:,} total  "
            f"{param_info['trainable']:,} trainable\n"
            f"  Det head  : {'YES' if model.det_head else 'NO'}\n"
            f"  Seg head  : {'YES' if model.seg_head else 'NO'}\n"
            f"  Device    : {device}"
        )