import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from distances import pairwise_distance


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al.).
    Pulls each example toward all same‐class embeddings, pushes away from others.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = temperature

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        # Compute scaled dot‐product similarities
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Numerical stability: subtract max on each row
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Positive‐pair mask and removal of self‐contrast
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(embeddings.device)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask

        # Exponentiate logits, excluding self‐contrast
        exp_logits = torch.exp(logits) * logits_mask

        # log‐prob for each pair: log( exp_logits / sum_exp_logits )
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-12))

        # For each anchor, average over its positive neighbours
        denom_pos = mask.sum(1).clamp(min=1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom_pos

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss 

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
