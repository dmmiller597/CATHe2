import torch
from torch import Tensor
from typing import Tuple

from .distances import pairwise_distance_optimized

class BatchHardMiner:
    """
    Implements Batch Hard Mining strategy for triplet selection within a batch.

    For each anchor, selects the hardest positive (farthest) and hardest negative
    (closest) sample within the batch based on squared Euclidean distance.
    """
    def __init__(self, distance_metric_func=pairwise_distance_optimized):
        self.distance_metric = distance_metric_func

    def __call__(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Selects batch-hard triplets.

        Args:
            embeddings: Tensor of embeddings (batch_size, embedding_dim). Assumed normalized.
            labels: Tensor of integer labels (batch_size,).

        Returns:
            A tuple containing the indices of (anchors, positives, negatives)
            that form valid hard triplets within the batch. Returns empty tensors
            if no valid triplets can be formed.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Calculate pairwise distances (squared Euclidean)
        dist_mat = self.distance_metric(embeddings, embeddings) # Shape: (batch_size, batch_size)

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1) # Shape: (batch_size, batch_size)
        labels_not_equal = ~labels_equal
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # --- Find Hardest Positive ---
        # Mask out self and negatives. For positives, we want the *max* distance.
        pos_dist_mat = dist_mat.clone()
        pos_dist_mat.masked_fill_(~labels_equal | identity_mask, -torch.inf) # Invalidate non-positives
        hardest_pos_dist, hardest_pos_idx = torch.max(pos_dist_mat, dim=1)

        # --- Find Hardest Negative ---
        # Mask out self and positives. For negatives, we want the *min* distance.
        neg_dist_mat = dist_mat.clone()
        # Invalidate positives and self by setting distance to +infinity
        neg_dist_mat.masked_fill_(labels_equal | identity_mask, torch.inf)
        hardest_neg_dist, hardest_neg_idx = torch.min(neg_dist_mat, dim=1)

        # --- Filter Valid Triplets ---
        # Valid if a positive exists (dist > -inf) and a negative exists (dist < inf)
        valid_pos_mask = hardest_pos_dist > -torch.inf
        valid_neg_mask = hardest_neg_dist < torch.inf
        valid_anchor_mask = valid_pos_mask & valid_neg_mask

        # Get the indices for the valid hard triplets
        anchor_indices = torch.where(valid_anchor_mask)[0]

        if len(anchor_indices) == 0:
            # log.debug("No valid triplets found in this batch.") # Can be noisy
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        positive_indices = hardest_pos_idx[anchor_indices]
        negative_indices = hardest_neg_idx[anchor_indices]

        return anchor_indices, positive_indices, negative_indices

class SemiHardMiner:
    """
    Implements Semi-Hard Mining strategy for triplet selection within a batch.

    For each anchor, selects the hardest positive (farthest) and semi-hard negative
    (further than the positive but not too far) sample using vectorized operations.
    """
    def __init__(self, distance_metric_func=pairwise_distance_optimized):
        self.distance_metric = distance_metric_func

    def __call__(self, embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Selects semi-hard triplets using vectorized operations.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Calculate pairwise distances (squared Euclidean)
        dist_mat = self.distance_metric(embeddings, embeddings)

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # --- Find Hardest Positive ---
        # Mask out self and negatives. For positives, we want the *max* distance.
        pos_dist_mat = dist_mat.clone()
        pos_dist_mat.masked_fill_(~labels_equal | identity_mask, -torch.inf)
        hardest_pos_dist, hardest_pos_idx = torch.max(pos_dist_mat, dim=1)

        # --- Find Closest-Harder Negative ---

        # 1. Get distance threshold for each anchor (based on its hardest positive)
        pos_dist_threshold = hardest_pos_dist.unsqueeze(1).expand(-1, batch_size) # Shape: (batch_size, batch_size)

        # 2. Create mask for negatives harder than the positive
        harder_negative_mask = (dist_mat > pos_dist_threshold)

        # 3. Only consider actual negatives (different class) and not self
        harder_negative_mask = harder_negative_mask & labels_not_equal & ~identity_mask

        # 4. Create negative distance matrix, invalidating non-harder negatives
        neg_dist_mat = dist_mat.clone()
        neg_dist_mat.masked_fill_(~harder_negative_mask, torch.inf) # Invalidate easy negatives

        # 5. Get the minimum distance negative (closest one that's harder than positive)
        # If no negative satisfies the condition for an anchor, its min distance will be inf
        hardest_neg_dist, hardest_neg_idx = torch.min(neg_dist_mat, dim=1)

        # --- Filter Valid Triplets ---
        # Valid if a positive exists (dist > -inf) AND a harder negative exists (dist < inf)
        valid_pos_mask = hardest_pos_dist > -torch.inf
        valid_neg_mask = hardest_neg_dist < torch.inf # Check if a harder negative was found
        valid_anchor_mask = valid_pos_mask & valid_neg_mask

        # Get the indices for the valid triplets
        anchor_indices = torch.where(valid_anchor_mask)[0]

        if len(anchor_indices) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        positive_indices = hardest_pos_idx[anchor_indices]
        negative_indices = hardest_neg_idx[anchor_indices]

        return anchor_indices, positive_indices, negative_indices 