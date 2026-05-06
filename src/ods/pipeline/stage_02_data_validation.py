import logging
from src.ods.components.data_validation import DataValidation
from src.ods.config.configuration import ConfigurationManager

logger = logging.getLogger(__name__)
STAGE = "Stage 02 — Data Validation"


def run(experiment: str = "exp_det_001") -> bool:
    logger.info(f"{'='*50}\n{STAGE}\n{'='*50}")
    cm = ConfigurationManager(experiment=experiment)
    cfg = cm.get_data_ingestion_config()
    validator = DataValidation(cfg)
    ok = validator.run_all_checks()
    if not ok:
        raise RuntimeError("Data validation failed. Fix dataset before training.")
    return ok


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "exp_det_001")