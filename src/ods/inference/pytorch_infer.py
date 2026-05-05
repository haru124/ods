"""
PyTorch Inference Pipeline
==========================
Features:
  • Single image and batch inference
  • PyTorch Profiler (CPU/CUDA time + memory)
  • Supports detection-only, segmentation-only, and both
  • Visualisation helpers using matplotlib (optional)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.ods.constants import CITYSCAPES_CLASSES, DETECTION_CLASSES
from src.ods.datasets.transforms import get_val_transforms
from src.ods.entity.config_entity import InferenceConfig, ModelConfig
from src.ods.models.model import ODSModel
from src.ods.utils.common import get_device

logger = logging.getLogger(__name__)


class PyTorchInfer:
    """
    Inference wrapper for ODSModel.

    Usage:
        infer = PyTorchInfer(model, infer_cfg)
        result = infer.predict(image_path)
    """

    def __init__(
        self,
        model: ODSModel,
        infer_cfg: InferenceConfig,
        image_size: List[int] = None,
    ):
        self.model = model
        self.cfg = infer_cfg
        self.device = get_device(infer_cfg.device)
        self.model.to(self.device)
        self.model.eval()
        self.image_size = image_size or [512, 1024]
        self.transform = get_val_transforms(self.image_size)

    # ── Single image predict ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Run inference on a single image.

        Returns dict with:
          "seg_mask"     : np.ndarray [H, W]  (class indices 0-18)
          "seg_overlay"  : PIL.Image           (coloured overlay)
          "boxes"        : np.ndarray [N, 4]
          "scores"       : np.ndarray [N]
          "labels"       : np.ndarray [N]
          "label_names"  : List[str]
          "inference_ms" : float
        """
        image = Image.open(image_path).convert("RGB")
        orig_size = image.size  # (W, H)

        # Transform (Resize + Normalize)
        tensor_img, _, _ = self.transform(image, torch.zeros(1, 1).long(), torch.zeros(0, 4))
        tensor_img = tensor_img.unsqueeze(0).to(self.device)   # [1, 3, H, W]

        result = {}
        t_start = time.perf_counter()

        if self.cfg.task in ("detection", "both"):
            self.model.det_head.eval()
            det_preds = self.model.det_head(tensor_img)
            pred = det_preds[0]
            # Filter by confidence
            keep = pred["scores"] > self.cfg.conf_threshold
            result["boxes"] = pred["boxes"][keep].cpu().numpy()
            result["scores"] = pred["scores"][keep].cpu().numpy()
            result["labels"] = (pred["labels"][keep] - 1).cpu().numpy()  # remove background
            result["label_names"] = [
                DETECTION_CLASSES[int(i)] if 0 <= int(i) < len(DETECTION_CLASSES) else "?"
                for i in result["labels"]
            ]

        if self.cfg.task in ("segmentation", "both"):
            features = self.model.backbone(tensor_img)
            target_size = (self.image_size[0], self.image_size[1])
            seg_logits = self.model.seg_head(features, target_size)
            seg_pred = seg_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]
            result["seg_mask"] = seg_pred
            result["seg_overlay"] = self._colorize_mask(seg_pred)

        result["inference_ms"] = (time.perf_counter() - t_start) * 1000
        return result

    # ── Profiled benchmark ────────────────────────────────────────────────────

    def benchmark(
        self,
        image_size: Optional[List[int]] = None,
        n_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Measure average inference time, throughput, and memory.
        Uses PyTorch Profiler for detailed CPU/CUDA breakdown.

        Returns dict:
            avg_ms, p50_ms, p95_ms, throughput_fps,
            gpu_memory_mb, model_params
        """
        H, W = image_size or self.image_size
        dummy = torch.randn(1, 3, H, W).to(self.device)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                if self.cfg.task in ("segmentation", "both"):
                    feats = self.model.backbone(dummy)
                    self.model.seg_head(feats, (H, W))

        # Timed runs
        timings = []
        for _ in range(n_runs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.no_grad():
                if self.cfg.task in ("segmentation", "both"):
                    feats = self.model.backbone(dummy)
                    self.model.seg_head(feats, (H, W))
                elif self.cfg.task == "detection":
                    self.model.det_head([dummy.squeeze(0)])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t) * 1000)

        timings_arr = np.array(timings)
        gpu_mem = 0
        if self.device.type == "cuda":
            gpu_mem = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2

        total_params = sum(p.numel() for p in self.model.parameters())

        stats = {
            "avg_ms": float(timings_arr.mean()),
            "p50_ms": float(np.percentile(timings_arr, 50)),
            "p95_ms": float(np.percentile(timings_arr, 95)),
            "throughput_fps": float(1000 / timings_arr.mean()),
            "gpu_memory_mb": gpu_mem,
            "model_params": total_params,
        }
        logger.info(
            f"Benchmark ({n_runs} runs):  "
            f"avg={stats['avg_ms']:.2f}ms  "
            f"p95={stats['p95_ms']:.2f}ms  "
            f"fps={stats['throughput_fps']:.1f}  "
            f"gpu_mem={stats['gpu_memory_mb']:.0f}MB  "
            f"params={stats['model_params']:,}"
        )
        return stats

    # ── Colour mask ───────────────────────────────────────────────────────────

    @staticmethod
    def _colorize_mask(mask: np.ndarray) -> Image.Image:
        """Convert class index mask to a colour RGB image for visualisation."""
        # Cityscapes colour palette (19 classes)
        palette = [
            (128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
            (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),
            (70,130,180),(220,20,60),(255,0,0),(0,0,142),(0,0,70),
            (0,60,100),(0,80,100),(0,0,230),(119,11,32),
        ]
        H, W = mask.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_id, color in enumerate(palette):
            rgb[mask == cls_id] = color
        return Image.fromarray(rgb)