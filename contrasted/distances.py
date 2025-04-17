# src/distances.py
import torch
from torch import Tensor

def pairwise_distance(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes the pairwise squared Euclidean distance between two sets of vectors

    Uses the formula: d(x, y)^2 = ||x||^2 - 2 * x^T * y + ||y||^2

    Args:
        x: Tensor of shape (N, D), where N is the number of vectors and D is the dimensionality.
        y: Tensor of shape (M, D), where M is the number of vectors and D is the dimensionality.

    Returns:
        Tensor of shape (N, M) containing the pairwise squared Euclidean distances.
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)

    dist = x_norm - 2.0 * torch.mm(x, y.transpose(0, 1)) + y_norm
    # Ensure non-negative distances due to floating point inaccuracies
    dist = torch.clamp(dist, min=0.0)
    return dist
