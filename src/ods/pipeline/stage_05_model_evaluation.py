import logging
import sys
from src.ods.components.model_evaluation import ModelEvaluator
from src.ods.components.data_ingestion import DataIngestion
from src.ods.config.configuration import ConfigurationManager
from src.ods.models.model import ODSModel

logger = logging.getLogger(__name__)
STAGE = "Stage 05 — Model Evaluation"


def run(
    experiment: str = "exp_det_001",
    checkpoint_path: str = None,
    split: str = "val",
) -> dict:
    logger.info(f"{'='*50}\n{STAGE}\n{'='*50}")
    cm = ConfigurationManager(experiment=experiment)
    data_cfg = cm.get_data_ingestion_config()
    model_cfg = cm.get_model_config()
    train_cfg = cm.get_training_config(experiment)
    eval_cfg = cm.get_evaluation_config()

    loaders = DataIngestion(data_cfg).get_dataloaders(task=train_cfg.task)

    model = ODSModel(model_cfg, task=train_cfg.task)
    ckpt = checkpoint_path or str(train_cfg.checkpoints_dir / "best_model.pth")
    model.load_weights(ckpt, strict=False)

    evaluator = ModelEvaluator(model, eval_cfg, train_cfg)
    return evaluator.evaluate(loaders[split], split=split)


if __name__ == "__main__":
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp_det_001"
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    split = sys.argv[3] if len(sys.argv) > 3 else "val"
    run(exp, ckpt, split)