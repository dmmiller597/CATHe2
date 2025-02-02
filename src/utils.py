import logging
import random
import numpy as np
import torch
from rich.logging import RichHandler
from typing import Optional

def set_seed(seed: int, workers: bool = True) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed
        workers: Whether to set seed for dataloader workers
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if workers:
        torch.use_deterministic_algorithms(True)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger with rich formatting.
    
    Args:
        name: Logger name (default: None)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )
    return logging.getLogger(name or "rich") 