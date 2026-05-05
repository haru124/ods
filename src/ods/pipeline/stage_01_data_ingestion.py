import logging
from src.ods.config.configuration import ConfigurationManager
from src.ods.datasets.cityscapes_dataset import get_dataloaders

logger = logging.getLogger(__name__)
STAGE = "Stage 01 — Data Ingestion"


def run(experiment: str = "exp_det_001", task: str = None) -> dict:
    logger.info(f"{'='*50}\n{STAGE}\n{'='*50}")
    cm = ConfigurationManager(experiment=experiment)
    cfg = cm.get_data_ingestion_config()
    train_cfg = cm.get_training_config(experiment)
    t = task or train_cfg.task
    loaders = get_dataloaders(cfg, task=t)
    logger.info(
        f"Loaders ready:  "
        f"train={len(loaders['train'].dataset)}  "
        f"val={len(loaders['val'].dataset)}  "
        f"test={len(loaders['test'].dataset)}"
    )
    return loaders


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp_det_001"
    run(exp)