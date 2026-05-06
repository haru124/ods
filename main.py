"""
main.py  —  ODS Project Entry Point
=====================================
Three modes:
  python main.py --task detection    --experiment exp_det_001
  python main.py --task segmentation --experiment exp_seg_001
  python main.py --task both         --experiment exp_both_001

Full pipeline (default):
  python main.py --experiment exp_seg_001

Skip stages:
  python main.py --experiment exp_seg_001 --skip-validation

Evaluate only:
  python main.py --experiment exp_seg_001 --eval-only --checkpoint path/to/best.pth

Infer on image:
  python main.py --experiment exp_seg_001 --infer --image path/img.png --checkpoint path/to/best.pth

Export ONNX:
  python main.py --experiment exp_seg_001 --export-onnx --checkpoint path/to/best.pth
"""

import argparse
import logging
import sys
from pathlib import Path

from src.ods.utils.common import setup_logging


def parse_args():
    p = argparse.ArgumentParser(description="ODS: Object Detection & Segmentation")

    p.add_argument("--experiment", default="exp_seg_001",
                   help="Experiment name from params.yaml")
    p.add_argument("--task",
                   choices=["detection", "segmentation", "both"],
                   default=None,
                   help="Override task from config")
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint for eval/infer/export")

    # Run mode
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    mode.add_argument("--infer", action="store_true")
    mode.add_argument("--export-onnx", action="store_true")

    # Flags
    p.add_argument("--skip-validation", action="store_true",
                   help="Skip dataset validation stage")
    p.add_argument("--image", default=None,
                   help="Image path for --infer mode")
    p.add_argument("--onnx-path", default=None,
                   help="ONNX path for inference comparison")
    p.add_argument("--benchmark", action="store_true",
                   help="Run inference benchmark")
    p.add_argument("--split", default="val",
                   choices=["val", "test"],
                   help="Split to evaluate on")
    return p.parse_args()


def run_full_pipeline(args):
    from src.ods.config.configuration import ConfigurationManager
    from src.ods.components.data_ingestion import DataIngestion
    from src.ods.components.data_validation import DataValidation
    from src.ods.components.model_builder import ModelBuilder
    from src.ods.components.model_trainer import Trainer
    from src.ods.components.model_evaluation import ModelEvaluator

    cm = ConfigurationManager(experiment=args.experiment)
    data_cfg = cm.get_data_ingestion_config()
    model_cfg = cm.get_model_config()
    eval_cfg = cm.get_evaluation_config()

    # Allow CLI task override on top of experiment config
    train_cfg = cm.get_training_config(args.experiment)
    if args.task:
        train_cfg.task = args.task

    logger = logging.getLogger(__name__)

    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STAGE 1 — Data Ingestion")
    ingestion = DataIngestion(data_cfg)
    loaders = ingestion.get_dataloaders(task=train_cfg.task)

    # ── Stage 2: Data Validation ──────────────────────────────────────────────
    if not args.skip_validation:
        logger.info("\n" + "="*60)
        logger.info("STAGE 2 — Data Validation")
        validator = DataValidation(data_cfg)
        validator.run_all_checks()

    # ── Stage 3: Model Builder ────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STAGE 3 — Model Builder")
    builder = ModelBuilder(model_cfg, train_cfg)
    model = builder.build(weight_path=args.checkpoint)

    if args.eval_only:
        # ── Eval only ─────────────────────────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("STAGE 5 — Evaluation Only")
        evaluator = ModelEvaluator(model, eval_cfg, train_cfg)
        evaluator.evaluate(loaders[args.split], split=args.split)
        return

    if not args.eval_only:
        # ── Stage 4: Training ─────────────────────────────────────────────────
        logger.info("\n" + "="*60)
        logger.info("STAGE 4 — Model Training")
        trainer = Trainer(model, train_cfg, model_cfg)
        trainer.fit(loaders["train"], loaders["val"])

    # ── Stage 5: Evaluation ───────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STAGE 5 — Post-Training Evaluation")
    evaluator = ModelEvaluator(model, eval_cfg, train_cfg)
    evaluator.evaluate(loaders["val"], split="val")


def run_infer(args):
    from src.ods.config.configuration import ConfigurationManager
    from src.ods.models.model import ODSModel
    from src.ods.inference.pytorch_infer import PyTorchInfer

    if not args.image:
        raise ValueError("--image is required for --infer mode")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --infer mode")

    cm = ConfigurationManager(experiment=args.experiment)
    model_cfg = cm.get_model_config()
    infer_cfg = cm.get_inference_config()
    data_cfg = cm.get_data_ingestion_config()
    if args.task:
        infer_cfg.task = args.task

    model = ODSModel(model_cfg, task=infer_cfg.task)
    model.load_weights(args.checkpoint)

    pt_infer = PyTorchInfer(model, infer_cfg, image_size=data_cfg.image_size)
    result = pt_infer.predict(args.image)

    print(f"\nInference time: {result['inference_ms']:.2f} ms")
    if "seg_mask" in result:
        print(f"Seg mask shape: {result['seg_mask'].shape}")
        if result.get("seg_overlay"):
            out_path = Path("artifacts") / "inference_output.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result["seg_overlay"].save(str(out_path))
            print(f"Overlay saved: {out_path}")
    if "boxes" in result:
        print(f"Detected {len(result['boxes'])} objects:")
        for box, score, name in zip(result["boxes"], result["scores"], result["label_names"]):
            print(f"  {name:15s}  score={score:.3f}  box={box.astype(int).tolist()}")

    if args.benchmark:
        pt_infer.benchmark(image_size=data_cfg.image_size)

    if args.onnx_path:
        from src.ods.inference.onnx_infer import ONNXInfer, compare_pytorch_vs_onnx
        onnx_infer = ONNXInfer(args.onnx_path, task=infer_cfg.task)
        compare_pytorch_vs_onnx(pt_infer, onnx_infer, args.image,
                                 image_size=tuple(data_cfg.image_size))


def run_export_onnx(args):
    from src.ods.config.configuration import ConfigurationManager
    from src.ods.models.model import ODSModel

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for --export-onnx")

    cm = ConfigurationManager(experiment=args.experiment)
    model_cfg = cm.get_model_config()
    infer_cfg = cm.get_inference_config()
    data_cfg = cm.get_data_ingestion_config()
    task = args.task or infer_cfg.task

    model = ODSModel(model_cfg, task=task)
    model.load_weights(args.checkpoint)

    onnx_dir = Path(infer_cfg.onnx_dir)
    onnx_dir.mkdir(parents=True, exist_ok=True)

    if task in ("segmentation", "both"):
        out = onnx_dir / f"{args.experiment}_seg.onnx"
        model.export_onnx(str(out), task="segmentation",
                          image_size=tuple(data_cfg.image_size))
    if task in ("detection", "both"):
        out = onnx_dir / f"{args.experiment}_det.onnx"
        model.export_onnx(str(out), task="detection",
                          image_size=tuple(data_cfg.image_size))


def main():
    args = parse_args()
    setup_logging(log_dir=Path("artifacts/logs"))
    logger = logging.getLogger(__name__)
    logger.info(f"ODS | experiment={args.experiment} | task={args.task or 'from config'}")

    try:
        if args.infer:
            run_infer(args)
        elif args.export_onnx:
            run_export_onnx(args)
        else:
            run_full_pipeline(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()