import logging
from src.ods.config.configuration import ConfigurationManager
from src.ods.models.model import ODSModel
from src.ods.utils.common import count_parameters

logger = logging.getLogger(__name__)
STAGE = "Stage 03 — Model Builder"


def run(experiment: str = "exp_det_001") -> ODSModel:
    logger.info(f"{'='*50}\n{STAGE}\n{'='*50}")
    cm = ConfigurationManager(experiment=experiment)
    model_cfg = cm.get_model_config()
    train_cfg = cm.get_training_config(experiment)
    model = ODSModel(model_cfg, task=train_cfg.task)
    count_parameters(model)
    return model


if __name__ == "__main__":
    import sys
    exp = sys.argv[1] if len(sys.argv) > 1 else "exp_det_001"
    run(exp)