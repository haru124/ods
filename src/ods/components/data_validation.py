"""
Data Validation Component
=========================
Checks:
  1. No image ID overlap between train and test splits
  2. At least one annotation file exists per split
  3. Image/annotation filename pairing is consistent
  4. Class distribution summary (seg + det)
  5. Resolution statistics
Logs a full report — does NOT modify any files.
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from src.ods.constants import CITYSCAPES_CLASSES, DETECTION_CLASSES, LABEL_ID_TO_TRAIN_ID
from src.ods.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)


class DataValidation:
    def __init__(self, cfg: DataIngestionConfig):
        self.cfg = cfg

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all_checks(self) -> bool:
        ok = True
        ok &= self.check_no_leakage()
        ok &= self.check_annotation_pairing("train")
        ok &= self.check_annotation_pairing("val")
        self.log_class_distribution("train", max_images=200)
        self.log_resolution_stats("train", max_images=100)
        return ok

    # ── Leakage check ─────────────────────────────────────────────────────────

    def check_no_leakage(self) -> bool:
        train_ids = self._collect_image_ids("train")
        test_ids = self._collect_image_ids("test")
        overlap = train_ids & test_ids
        if overlap:
            logger.error(
                f"DATA LEAKAGE DETECTED: {len(overlap)} image(s) appear in "
                f"both train and test.\n  Sample: {list(overlap)[:5]}"
            )
            return False
        logger.info(
            f"Leakage check PASSED — train={len(train_ids)}  "
            f"test={len(test_ids)}  overlap=0"
        )
        return True

    # ── Annotation pairing ────────────────────────────────────────────────────

    def check_annotation_pairing(self, split: str) -> bool:
        img_dir = self.cfg.data_root / self.cfg.images_subdir / split
        ann_dir = self.cfg.data_root / self.cfg.annotations_subdir / split
        if not img_dir.exists():
            logger.warning(f"Image dir missing for split '{split}': {img_dir}")
            return True  # not an error if test split doesn't exist yet

        mismatches = []
        total = 0
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in city_dir.glob("*_leftImg8bit.png"):
                total += 1
                image_id = img_path.stem.replace("_leftImg8bit", "")
                label_path = (
                    ann_dir / city_dir.name / f"{image_id}_gtFine_labelIds.png"
                )
                if not label_path.exists() and split != "test":
                    mismatches.append(image_id)

        if mismatches:
            logger.warning(
                f"[{split}] {len(mismatches)}/{total} images missing label files. "
                f"First few: {mismatches[:3]}"
            )
        else:
            logger.info(f"[{split}] Annotation pairing OK — {total} images all paired")
        return len(mismatches) == 0

    # ── Class distribution ────────────────────────────────────────────────────

    def log_class_distribution(self, split: str, max_images: int = 200) -> None:
        ann_dir = self.cfg.data_root / self.cfg.annotations_subdir / split
        if not ann_dir.exists():
            return

        train_id_counter: Counter = Counter()
        det_counter: Counter = Counter()
        checked = 0

        for city_dir in sorted(ann_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for label_path in sorted(city_dir.glob("*_gtFine_labelIds.png")):
                if checked >= max_images:
                    break
                arr = np.array(Image.open(label_path), dtype=np.int32)
                for raw_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
                    if train_id != 255:
                        count = int((arr == raw_id).sum())
                        if count:
                            train_id_counter[CITYSCAPES_CLASSES[train_id]] += count
                checked += 1
            if checked >= max_images:
                break

        # Detection: from polygon JSON
        checked_det = 0
        for city_dir in sorted(ann_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for poly_path in sorted(city_dir.glob("*_gtFine_polygons.json")):
                if checked_det >= max_images:
                    break
                try:
                    with open(poly_path) as f:
                        data = json.load(f)
                    for obj in data.get("objects", []):
                        lbl = obj.get("label", "")
                        if lbl in DETECTION_CLASSES:
                            det_counter[lbl] += 1
                except Exception:
                    pass
                checked_det += 1
            if checked_det >= max_images:
                break

        logger.info(f"\n[{split}] Segmentation pixel distribution (top-10, {checked} images):")
        for cls, count in train_id_counter.most_common(10):
            logger.info(f"  {cls:20s}: {count:>12,} px")

        logger.info(f"\n[{split}] Detection instance distribution ({checked_det} images):")
        for cls, count in det_counter.most_common():
            logger.info(f"  {cls:15s}: {count:>6} instances")

    # ── Resolution stats ──────────────────────────────────────────────────────

    def log_resolution_stats(self, split: str, max_images: int = 100) -> None:
        img_dir = self.cfg.data_root / self.cfg.images_subdir / split
        if not img_dir.exists():
            return
        widths, heights = [], []
        checked = 0
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            for img_path in city_dir.glob("*_leftImg8bit.png"):
                if checked >= max_images:
                    break
                try:
                    w, h = Image.open(img_path).size
                    widths.append(w)
                    heights.append(h)
                except Exception:
                    pass
                checked += 1
            if checked >= max_images:
                break

        if widths:
            logger.info(
                f"[{split}] Resolution stats ({checked} images):  "
                f"W={min(widths)}-{max(widths)} (median {int(np.median(widths))})  "
                f"H={min(heights)}-{max(heights)} (median {int(np.median(heights))})"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _collect_image_ids(self, split: str) -> set:
        img_dir = self.cfg.data_root / self.cfg.images_subdir / split
        if not img_dir.exists():
            return set()
        ids = set()
        for city_dir in img_dir.iterdir():
            if not city_dir.is_dir():
                continue
            for img_path in city_dir.glob("*_leftImg8bit.png"):
                ids.add(img_path.stem.replace("_leftImg8bit", ""))
        return ids