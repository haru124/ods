"""
Prediction Pipeline
===================
Supports three run modes via --task argument:
  detection    — run detection head only
  segmentation — run segmentation head only
  both         — run both heads

Usage:
  python -m src.ods.pipeline.prediction_pipeline \\
         --image path/to/image.png \\
         --checkpoint artifacts/checkpoints/exp_seg_001/best_model.pth \\
         --task segmentation \\
         --experiment exp_seg_001
"""

import argparse
import logging
from pathlib import Path

from src.ods.config.configuration import ConfigurationManager
from src.ods.inference.pytorch_infer import PyTorchInfer
from src.ods.models.model import ODSModel
from src.ods.utils.common import setup_logging

logger = logging.getLogger(__name__)


def predict(
    image_path: str,
    checkpoint_path: str,
    experiment: str = "exp_seg_001",
    task: str = None,
    onnx_path: str = None,
    benchmark: bool = False,
) -> dict:
    setup_logging()
    cm = ConfigurationManager(experiment=experiment)
    model_cfg = cm.get_model_config()
    infer_cfg = cm.get_inference_config()
    data_cfg = cm.get_data_ingestion_config()

    # Override task if provided
    if task:
        infer_cfg.task = task
        model_task = task
    else:
        model_task = infer_cfg.task

    # Build and load model
    model = ODSModel(model_cfg, task=model_task)
    model.load_weights(checkpoint_path)

    # PyTorch inference
    pt_infer = PyTorchInfer(model, infer_cfg, image_size=data_cfg.image_size)
    result = pt_infer.predict(image_path)
    logger.info(f"PyTorch inference: {result['inference_ms']:.2f} ms")

    if benchmark:
        pt_stats = pt_infer.benchmark(image_size=data_cfg.image_size)
        result["benchmark"] = pt_stats

    # ONNX comparison (optional)
    if onnx_path and Path(onnx_path).exists():
        from src.ods.inference.onnx_infer import ONNXInfer, compare_pytorch_vs_onnx
        onnx_infer = ONNXInfer(onnx_path, task=model_task)
        comparison = compare_pytorch_vs_onnx(
            pt_infer, onnx_infer, image_path,
            image_size=tuple(data_cfg.image_size),
        )
        result["onnx_comparison"] = comparison

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", default="exp_seg_001")
    parser.add_argument("--task", choices=["detection", "segmentation", "both"], default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    result = predict(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        experiment=args.experiment,
        task=args.task,
        onnx_path=args.onnx,
        benchmark=args.benchmark,
    )
    import json
    # Print summary (skip large arrays)
    summary = {k: v for k, v in result.items()
               if not isinstance(v, type(None)) and k not in ("seg_mask", "seg_overlay")}
    print(json.dumps(summary, indent=2, default=str))