import torch
import torch.nn.functional as F
from torch import Tensor


from distances import pairwise_distance

def soft_triplet_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor,
    distance_metric_func=pairwise_distance
) -> Tensor:
    """
    Computes the soft triplet loss using the softplus function.

    Loss = log(1 + exp(d(anchor, positive)^2 - d(anchor, negative)^2))
    Aims to push d(a, p) down and d(a, n) up.

    Args:
        anchor: Embeddings for anchor samples.
        positive: Embeddings for positive samples.
        negative: Embeddings for negative samples.
        distance_metric_func: Function to compute pairwise distances.

    Returns:
        The mean soft triplet loss over the batch.
    """
    # Note: assumes distance_metric_func takes (x, y) and returns matrix
    # We need pairwise distances between corresponding anchor/positive and anchor/negative pairs
    # If anchor, positive, negative are (N, dim), we want N distances, not N*N
    # For cdist based metrics, we need to compute full matrices and take the diagonal

    # Calculate distances between each anchor and its corresponding positive/negative
    # Efficiently using diagonal after full pairwise calculation
    dist_ap_full = distance_metric_func(anchor, positive) # Shape (N, N)
    dist_an_full = distance_metric_func(anchor, negative) # Shape (N, N)

    # Extract the diagonal elements which represent the paired distances
    dist_ap = dist_ap_full.diag()
    dist_an = dist_an_full.diag()


    # Softplus(x) = log(1 + exp(x))
    loss = F.softplus(dist_ap - dist_an)
    return loss.mean() 