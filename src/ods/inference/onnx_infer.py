"""
ONNX Inference Pipeline
========================
Loads the exported ONNX model and runs inference with onnxruntime.
Benchmarks and compares against PyTorch.

Teaching note — what ONNX is:
  ONNX (Open Neural Network Exchange) is an open format for ML models.
  It serialises the model graph (ops + weights) into a .onnx file.
  onnxruntime (ORT) then executes this graph — often faster than PyTorch
  because ORT does graph-level optimisations (op fusion, memory planning).
  Key benefit: deploy on any hardware without a Python/PyTorch install.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src.ods.datasets.transforms import IMAGENET_MEAN, IMAGENET_STD

logger = logging.getLogger(__name__)


class ONNXInfer:
    """
    Inference using ONNX Runtime.

    Usage:
        infer = ONNXInfer("artifacts/onnx/seg_model.onnx", task="segmentation")
        result = infer.predict(image_path)
        stats = infer.benchmark()
    """

    def __init__(
        self,
        onnx_path: str,
        task: str = "segmentation",
        use_gpu: bool = False,
    ):
        import onnxruntime as ort

        self.onnx_path = str(onnx_path)
        self.task = task

        # Session options
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            f"ONNX session loaded: {self.onnx_path}  "
            f"providers={self.session.get_providers()}"
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    @staticmethod
    def preprocess(image_path: str, image_size: Tuple[int, int] = (512, 1024)) -> np.ndarray:
        """PIL → normalised float32 numpy array [1, 3, H, W]."""
        img = Image.open(image_path).convert("RGB").resize(
            (image_size[1], image_size[0]), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)             # [3, H, W]
        return arr[np.newaxis, :, :, :]          # [1, 3, H, W]

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, image_path: str, image_size: Tuple[int, int] = (512, 1024)) -> Dict:
        input_arr = self.preprocess(image_path, image_size)
        t = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_arr})
        elapsed_ms = (time.perf_counter() - t) * 1000

        result = {"inference_ms": elapsed_ms}
        if self.task == "segmentation":
            logits = outputs[0]                  # [1, C, H, W]
            seg_mask = logits.argmax(axis=1).squeeze(0)  # [H, W]
            result["seg_mask"] = seg_mask
        elif self.task == "detection":
            result["backbone_features"] = outputs[0]

        return result

    # ── Benchmark ─────────────────────────────────────────────────────────────

    def benchmark(
        self,
        image_size: Tuple[int, int] = (512, 1024),
        n_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Measure average ORT inference time.
        Compare results with PyTorchInfer.benchmark() to see the speedup.
        """
        dummy = np.random.randn(1, 3, *image_size).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            self.session.run(None, {self.input_name: dummy})

        timings = []
        for _ in range(n_runs):
            t = time.perf_counter()
            self.session.run(None, {self.input_name: dummy})
            timings.append((time.perf_counter() - t) * 1000)

        timings_arr = np.array(timings)
        stats = {
            "avg_ms": float(timings_arr.mean()),
            "p50_ms": float(np.percentile(timings_arr, 50)),
            "p95_ms": float(np.percentile(timings_arr, 95)),
            "throughput_fps": float(1000 / timings_arr.mean()),
        }
        logger.info(
            f"ONNX Benchmark ({n_runs} runs):  "
            f"avg={stats['avg_ms']:.2f}ms  "
            f"fps={stats['throughput_fps']:.1f}"
        )
        return stats


def compare_pytorch_vs_onnx(
    pytorch_infer,
    onnx_infer: ONNXInfer,
    image_path: str,
    image_size: Tuple[int, int] = (512, 1024),
    n_runs: int = 100,
) -> Dict:
    """
    Full comparison: speed + output consistency.

    Output consistency check: cosine similarity between seg logits
    from PyTorch and ONNX — should be > 0.999 for a correct export.
    """
    pt_stats = pytorch_infer.benchmark(image_size=list(image_size), n_runs=n_runs)
    onnx_stats = onnx_infer.benchmark(image_size=image_size, n_runs=n_runs)

    speedup = pt_stats["avg_ms"] / onnx_stats["avg_ms"]

    # Output consistency
    pt_result = pytorch_infer.predict(image_path)
    onnx_result = onnx_infer.predict(image_path, image_size)

    consistency = None
    if "seg_mask" in pt_result and "seg_mask" in onnx_result:
        pt_mask = pt_result["seg_mask"].astype(np.float32).ravel()
        onnx_mask = onnx_result["seg_mask"].astype(np.float32).ravel()
        # Simple agreement: fraction of pixels with same class
        consistency = float((pt_mask == onnx_mask).mean())

    report = {
        "pytorch_avg_ms": pt_stats["avg_ms"],
        "onnx_avg_ms": onnx_stats["avg_ms"],
        "speedup_x": speedup,
        "pytorch_fps": pt_stats["throughput_fps"],
        "onnx_fps": onnx_stats["throughput_fps"],
        "output_agreement": consistency,
        "pytorch_params": pt_stats.get("model_params", 0),
        "pytorch_gpu_mb": pt_stats.get("gpu_memory_mb", 0),
    }
    logger.info(
        f"\nPyTorch vs ONNX\n"
        f"  PyTorch : {pt_stats['avg_ms']:.2f}ms  ({pt_stats['throughput_fps']:.1f} fps)\n"
        f"  ONNX    : {onnx_stats['avg_ms']:.2f}ms  ({onnx_stats['throughput_fps']:.1f} fps)\n"
        f"  Speedup : {speedup:.2f}x\n"
        f"  Agreement: {consistency:.4f}" if consistency else ""
    )
    return report