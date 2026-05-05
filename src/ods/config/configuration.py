from pathlib import Path
from typing import Dict, Optional

from src.ods.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.ods.entity.config_entity import (
    DataIngestionConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from src.ods.utils.common import read_yaml


class ConfigurationManager:
    """
    Reads config.yaml + params.yaml and builds typed config objects.

    Usage:
        cm = ConfigurationManager(experiment="exp_det_001")
        data_cfg = cm.get_data_ingestion_config()
        model_cfg = cm.get_model_config()
        train_cfg = cm.get_training_config()
    """

    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH,
        experiment: Optional[str] = None,
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)

        # Merge experiment-level overrides on top of base config
        self.exp_params: Dict = {}
        if experiment:
            if experiment not in self.params.get("experiments", {}):
                raise ValueError(
                    f"Experiment '{experiment}' not found in params.yaml. "
                    f"Available: {list(self.params['experiments'].keys())}"
                )
            self.exp_params = self.params["experiments"][experiment]

        # Shortcut accessors
        self._paths = self.config["paths"]
        self._dataset = self.config["dataset"]
        self._dl = self.config["dataloader"]
        self._model = self.config["model"]
        self._training = self.config["training"]
        self._evaluation = self.config["evaluation"]
        self._inference = self.config["inference"]

    def _override(self, base_key: str, default=None):
        """Return experiment param if it exists, else fall back to base config."""
        return self.exp_params.get(base_key, default)

    # ── Data ──────────────────────────────────────────────────────────────────

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            data_root=Path(self._paths["data_root"]),
            images_subdir=self._dataset["images_subdir"],
            annotations_subdir=self._dataset["annotations_subdir"],
            image_size=self._dataset["image_size"],
            num_classes_seg=self._dataset["num_classes_seg"],
            num_classes_det=self._dataset["num_classes_det"],
            ignore_index=self._dataset["ignore_index"],
            train_batch_size=self._dl["train_batch_size"],
            val_batch_size=self._dl["val_batch_size"],
            num_workers=self._dl["num_workers"],
            pin_memory=self._dl["pin_memory"],
        )

    # ── Model ─────────────────────────────────────────────────────────────────

    def get_model_config(self) -> ModelConfig:
        m = self._model
        det = m["detection"]
        seg = m["segmentation"]
        return ModelConfig(
            backbone=self._override("backbone") or m["backbone"],
            pretrained_backbone=m["pretrained_backbone"],
            fpn_out_channels=m["fpn_out_channels"],
            anchor_sizes=det["anchor_sizes"],
            anchor_ratios=det["anchor_ratios"],
            rpn_fg_iou_thresh=det["rpn_fg_iou_thresh"],
            rpn_bg_iou_thresh=det["rpn_bg_iou_thresh"],
            box_score_thresh=det["box_score_thresh"],
            box_nms_thresh=det["box_nms_thresh"],
            box_detections_per_img=det["box_detections_per_img"],
            num_classes_det=self._dataset["num_classes_det"],
            decoder=self._override("decoder") or seg["decoder"],
            decoder_channels=seg["decoder_channels"],
            dropout=seg["dropout"],
            num_classes_seg=self._dataset["num_classes_seg"],
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def get_training_config(self, experiment: str = "default") -> TrainingConfig:
        tr = self._training
        return TrainingConfig(
            task=self._override("task") or self.config["project"]["task"],
            epochs=self._override("epochs") or tr["epochs"],
            optimizer=self._override("optimizer") or tr["optimizer"],
            lr=self._override("lr") or tr["lr"],
            momentum=tr["momentum"],
            weight_decay=tr["weight_decay"],
            scheduler=self._override("scheduler") or tr["scheduler"],
            warmup_epochs=tr["warmup_epochs"],
            detection_loss=self._override("detection_loss") or tr["detection_loss"],
            segmentation_loss=self._override("segmentation_loss") or tr["segmentation_loss"],
            clip_grad_norm=tr["clip_grad_norm"],
            use_amp=tr["use_amp"],
            resume_checkpoint=tr["resume_checkpoint"],
            checkpoints_dir=Path(self._paths["checkpoints_dir"]) / experiment,
            logs_dir=Path(self._paths["logs_dir"]) / experiment,
            mlflow_tracking_uri=self._paths["mlflow_tracking_uri"],
            experiment_name=experiment,
            description=self.exp_params.get("description", ""),
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def get_evaluation_config(self) -> EvaluationConfig:
        ev = self._evaluation
        return EvaluationConfig(
            det_iou_thresholds=ev["det_iou_thresholds"],
            seg_metrics=ev["seg_metrics"],
            eval_every_n_epochs=ev["eval_every_n_epochs"],
            num_classes_seg=self._dataset["num_classes_seg"],
            num_classes_det=self._dataset["num_classes_det"],
            ignore_index=self._dataset["ignore_index"],
        )

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_inference_config(self) -> InferenceConfig:
        inf = self._inference
        return InferenceConfig(
            device=inf["device"],
            conf_threshold=inf["conf_threshold"],
            nms_threshold=inf["nms_threshold"],
            profile_runs=inf["profile_runs"],
            warmup_runs=inf["warmup_runs"],
            checkpoints_dir=Path(self._paths["checkpoints_dir"]),
            onnx_dir=Path(self._paths["onnx_dir"]),
            task=self._override("task") or self.config["project"]["task"],
        )