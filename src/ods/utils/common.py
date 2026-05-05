import os
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

logger = logging.getLogger(__name__)


# ── YAML helpers ──────────────────────────────────────────────────────────────

def read_yaml(path: Path) -> Dict:
    """Load a YAML file and return as dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config: {path}")
    return cfg


def save_yaml(data: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


# ── Directory helpers ─────────────────────────────────────────────────────────

def create_directories(paths: list) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {p}")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    state: Dict,
    checkpoint_dir: Path,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(state, best_path)
        logger.info(f"New best model saved: {best_path}")


def load_checkpoint(path: str, device: torch.device) -> Dict:
    """Load a checkpoint, mapping storage to the correct device."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    logger.info(f"Loaded checkpoint: {path}  |  epoch={checkpoint.get('epoch', '?')}")
    return checkpoint


# ── Device helper ─────────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    elif preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(log_dir) / "run.log"))
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ── Timing utility ────────────────────────────────────────────────────────────

class Timer:
    """Simple context-manager timer."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.elapsed*1000:.2f} ms"


# ── JSON helpers ──────────────────────────────────────────────────────────────

def save_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


# ── Model size helper ─────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: total={total:,}  trainable={trainable:,}")
    return {"total": total, "trainable": trainable}