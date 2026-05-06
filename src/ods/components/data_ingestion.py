"""
Data Ingestion Component
========================
Responsible for:
  - Verifying dataset folder structure exists
  - Counting samples per split
  - Collecting train image IDs (used downstream to guard test split)
  - Returning DataLoaders ready for training
"""

import logging
from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from src.ods.datasets.cityscapes_dataset import get_dataloaders
from src.ods.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, cfg: DataIngestionConfig):
        self.cfg = cfg

    def validate_structure(self) -> bool:
        """Check required folders exist before attempting to build datasets."""
        required = [
            self.cfg.data_root / self.cfg.images_subdir / split
            for split in ("train", "val")
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            logger.error(f"Missing dataset directories:\n  " + "\n  ".join(missing))
            logger.error(
                "Expected layout:\n"
                "  data/cityscapes/Images/{train,val,test}/<city>/<img>.png\n"
                "  data/cityscapes/gtFine/{train,val,test}/<city>/<ann>"
            )
            return False
        logger.info("Dataset structure validated OK")
        return True

    def get_dataloaders(self, task: str = "both") -> Dict[str, DataLoader]:
        if not self.validate_structure():
            raise FileNotFoundError(
                "Dataset not found. Check data/cityscapes/ layout."
            )
        loaders = get_dataloaders(self.cfg, task=task)
        for split, loader in loaders.items():
            logger.info(
                f"  {split:5s}: {len(loader.dataset):5d} samples  "
                f"batch={loader.batch_size}"
            )
        return loaders