import logging
import os
import torch
import lightning as L
from rich.logging import RichHandler
from typing import Optional

_logging_configured = False

def configure_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure logging once with RichHandler and return a logger."""
    global _logging_configured
    if not _logging_configured:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        _logging_configured = True
    return logging.getLogger(name or __name__)

def set_seed(seed: int, workers: bool = True, deterministic: bool = True) -> None:
    """Set random seed for reproducibility and enforce deterministic behavior."""
    L.seed_everything(seed, workers=workers)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with rich formatting."""
    return configure_logging(name=name) 