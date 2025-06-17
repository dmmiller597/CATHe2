# src/distances.py
import torch
from torch import Tensor
import torch.nn.functional as F

def pairwise_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the pairwise L2 distance between two tensors of embeddings.
    Uses torch.cdist for efficiency.

    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)

    Returns:
        Tensor of shape (N, M) containing pairwise L2 distances.
    """
    return torch.cdist(x, y, p=2.0)


def pairwise_cosine_similarity(X: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise cosine similarity between row vectors in a tensor.
    Assumes embeddings are already L2-normalized.
    """
    # Embeddings are assumed to be normalized, so we can just do a matrix multiplication.
    return torch.matmul(X, X.T)
