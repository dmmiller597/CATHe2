import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from distances import pairwise_distance, pairwise_cosine_similarity
from utils import pairwise_jaccard_similarity, pairwise_overlap_coefficient


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
    

class JaccardLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, embeddings, labels):
        # compute the pairwise jaccard similarity between all the labels in a batch
        jaccs = torch.tensor(
            pairwise_jaccard_similarity(labels), device=embeddings.device
        )
        cosines = torch.abs(pairwise_cosine_similarity(embeddings))
        lower_indices = torch.tril_indices(
            cosines.shape[0], cosines.shape[1], offset=-1,
            device=cosines.device,
        )
        lower_jaccs = jaccs[lower_indices[0], lower_indices[1]].reshape(-1, 1)
        lower_cosines = torch.clip(cosines[lower_indices[0], lower_indices[1]].reshape(-1, 1), 0, 1)
        return torch.nn.functional.mse_loss(lower_cosines.float(), lower_jaccs)


class OverlapLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(self, embeddings, labels):
        overlap = torch.tensor(
            # The matrix size is implicitly determined by the length of the labels list.
            pairwise_overlap_coefficient(labels),
            device=embeddings.device,
            dtype=embeddings.dtype,
        )  # compute the pairwise overlap coefficient between all the labels in a batch with correct shape
        cosines = pairwise_cosine_similarity(embeddings)
        lower_indices = torch.tril_indices(
            cosines.shape[0], cosines.shape[1], offset=-1
        )
        lower_overlap = overlap[lower_indices].reshape(-1, 1)
        lower_cosines = torch.clip(cosines[lower_indices].reshape(-1, 1), 0, 1)
        return torch.nn.functional.mse_loss(lower_cosines, lower_overlap)


class SINCERELoss(nn.Module):
    """
    Supervised InfoNCE REvisited loss with cosine distance. 
    https://github.com/tufts-ml/SupContrast/blob/master/revised_losses.py
    """
    def __init__(self, temperature=0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeds: torch.Tensor, labels: torch.tensor):
        """Supervised InfoNCE REvisited loss with cosine distance

        Args:
            embeds (torch.Tensor): (B, D) embeddings of B images normalized over D dimension.
            labels (torch.tensor): (B,) integer class labels.

        Returns:
            torch.Tensor: Scalar loss.
        """
        # calculate logits (activations) for each embeddings pair (B, B)
        # using matrix multiply instead of cosine distance function for ~10x cost reduction
        logits = embeds @ embeds.T
        logits /= self.temperature
        # Cast logits to float32 to ensure stable calculations with mixed precision
        logits = logits.float()
        
        # determine which logits are between embeds of the same label (B, B)
        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)

        # masking with -inf to get zeros in the summation for the softmax denominator
        # Ensure operations using logits now use float32
        denom_activations = torch.full_like(logits, float("-inf"), dtype=torch.float32) 
        denom_activations[~same_label] = logits[~same_label]
        # get logsumexp of the logits between embeds of different labels for each row (B,)
        base_denom_row = torch.logsumexp(denom_activations, dim=0)
        # reshape to be (B, B) with row values equivalent, to be masked later
        base_denom = base_denom_row.unsqueeze(1).repeat((1, len(base_denom_row)))

        # get mask for numerator terms by removing comparisons between an image and itself (B, B)
        in_numer = same_label
        in_numer[torch.eye(in_numer.shape[0], dtype=bool)] = False
        # delete same_label so don't need to copy for in_numer
        del same_label
        # count numerator terms for averaging (B,)
        numer_count = in_numer.sum(dim=0).clamp(min=1)
        # numerator activations with others zeroed (B, B)
        # Ensure operations using logits now use float32
        numer_logits = torch.zeros_like(logits, dtype=torch.float32)
        numer_logits[in_numer] = logits[in_numer]

        # construct denominator term for each numerator via logsumexp over a stack (B, B)
        # Ensure operations using logits now use float32
        log_denom = torch.zeros_like(logits, dtype=torch.float32)
        # Stacking should now work as inputs are float32
        log_denom[in_numer] = torch.stack(
            (numer_logits[in_numer], base_denom[in_numer]), dim=0).logsumexp(dim=0)

        # cross entropy loss of each positive pair with the logsumexp of the negative classes (B, B)
        # entries not in numerator set to 0
        ce = -1 * (numer_logits - log_denom)
        # take average over rows with entry count then average over batch
        loss = torch.sum(ce / numer_count) / ce.shape[0]
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
