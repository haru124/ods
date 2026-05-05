import logging
from src.ods.config.configuration import ConfigurationManager
from src.ods.components.model_trainer import Trainer
from src.ods.datasets.cityscapes_dataset import get_dataloaders
from src.ods.models.model import ODSModel

logger = logging.getLogger(__name__)
STAGE = "Stage 04 — Model Trainer"


def run(experiment: str = "exp_det_001", model: ODSModel = None) -> None:
    logger.info(f"{'='*50}\n{STAGE}\n{'='*50}")
    cm = ConfigurationManager(experiment=experiment)
    data_cfg = cm.get_data_ingestion_config()
    model_cfg = cm.get_model_config()
    train_cfg = cm.get_training_config(experiment)

    loaders = get_dataloaders(data_cfg, task=train_cfg.task)

    if model is None:
        model = ODSModel(model_cfg, task=train_cfg.task)

    trainer = Trainer(model, train_cfg, model_cfg)
    trainer.fit(loaders["train"], loaders["val"])


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp_det_001"
    run(exp)