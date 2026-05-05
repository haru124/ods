from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataIngestionConfig:
    data_root: Path
    images_subdir: str
    annotations_subdir: str
    image_size: List[int]            # [H, W]
    num_classes_seg: int
    num_classes_det: int
    ignore_index: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class ModelConfig:
    backbone: str                     # resnet18 | resnet34 | resnet50
    pretrained_backbone: bool
    fpn_out_channels: int
    # Detection
    anchor_sizes: List[List[int]]
    anchor_ratios: List[float]
    rpn_fg_iou_thresh: float
    rpn_bg_iou_thresh: float
    box_score_thresh: float
    box_nms_thresh: float
    box_detections_per_img: int
    num_classes_det: int
    # Segmentation
    decoder: str                      # fcn | deeplab
    decoder_channels: int
    dropout: float
    num_classes_seg: int


@dataclass
class TrainingConfig:
    task: str                         # detection | segmentation | both
    epochs: int
    optimizer: str
    lr: float
    momentum: float
    weight_decay: float
    scheduler: str
    warmup_epochs: int
    detection_loss: str
    segmentation_loss: str
    clip_grad_norm: float
    use_amp: bool
    resume_checkpoint: str
    checkpoints_dir: Path
    logs_dir: Path
    mlflow_tracking_uri: str
    experiment_name: str
    description: str


@dataclass
class EvaluationConfig:
    det_iou_thresholds: List[float]
    seg_metrics: List[str]
    eval_every_n_epochs: int
    num_classes_seg: int
    num_classes_det: int
    ignore_index: int


@dataclass
class InferenceConfig:
    device: str
    conf_threshold: float
    nms_threshold: float
    profile_runs: int
    warmup_runs: int
    checkpoints_dir: Path
    onnx_dir: Path
    task: str