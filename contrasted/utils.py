import logging
import os
import torch
import lightning as L
from rich.logging import RichHandler
from typing import Optional, List
import random
import numpy as np

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

def convert_sf_string_to_list(sf_string: str) -> List[str]:
    """
    Args:
        sf_string: SF number represented as a string.
            e.g. '1.2.3.4'

    Returns:
        Taking the above example,
        '1.2.3.4' goes to
        ['1.', '1.2.', '1.2.3.', '1.2.3.4.']
    """
    parts = sf_string.split(".")
    return [".".join(parts[: i + 1]) + "." for i in range(len(parts))]

def jaccard_similarity(list1: list, list2: list) -> float:
    """
    Compute the Jaccard similarity between two sets.
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def pairwise_jaccard_similarity(
    list_of_lists: List[List[str]],
) -> List[List[float]]:
    """
    Compute the pairwise Jaccard similarity between
    all sets within a big list.
    """
    n = len(list_of_lists)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            sim = jaccard_similarity(list_of_lists[i], list_of_lists[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix

def overlap_coefficient(list1: list, list2: list) -> float:
    """
    Compute the Overlap coefficient between two sets.
    """
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))
    return intersection / min_size if min_size > 0 else 0.0

def pairwise_overlap_coefficient(
    list_of_lists: List[List[str]],
    embeddings: Optional[torch.Tensor] = None,
) -> List[List[float]]:
    """
    Compute the pairwise overlap coefficient
    between all sets within a big list of lists.
    """
    n = embeddings.shape[0] if embeddings is not None else len(list_of_lists)
    matrix = [[0.0] * n for _ in range(n)]
    # Use original list length for loop bounds to avoid index errors on list_of_lists
    # if DistributedSampler added padding.
    effective_length = len(list_of_lists)
    for i in range(effective_length):
        for j in range(i, effective_length):
            sim = overlap_coefficient(list_of_lists[i], list_of_lists[j])
            matrix[i][j] = sim
            matrix[j][i] = sim
    return matrix 