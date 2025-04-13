import logging
import time
import warnings # Import the warnings module
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor # Explicit type hinting
import os
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau # Explicit import
from .plotting import generate_tsne_plot, generate_umap_plot

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

# --- Distance & Mining ---

def pairwise_distance_optimized(x: Tensor, y: Tensor) -> Tensor:
    """
    Computes squared pairwise Euclidean distances between two batches of vectors
    using torch.cdist for efficiency.
    """
    return torch.cdist(x, y, p=2.0).pow(2)

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

# --- Loss Functions ---

def soft_triplet_loss(
    anchor: Tensor, positive: Tensor, negative: Tensor,
    distance_metric_func=pairwise_distance_optimized
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
    dist_ap = distance_metric_func(anchor, positive).diag() # Get diagonal for paired distances
    dist_an = distance_metric_func(anchor, negative).diag() # Get diagonal for paired distances

    # Softplus(x) = log(1 + exp(x))
    loss = F.softplus(dist_ap - dist_an)
    return loss.mean()

# --- Main Lightning Module ---

class ContrastiveCATHeModel(pl.LightningModule):
    """
    PyTorch Lightning module for CATH superfamily contrastive learning.

    Uses a projection head, Batch Hard Triplet Loss, and kNN evaluation.
    Assumes input is pre-computed protein embeddings.
    """
    def __init__(
        self,
        input_embedding_dim: int,
        projection_hidden_dims: List[int] = [1024],
        output_embedding_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        use_layer_norm: bool = True,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        knn_val_neighbors: int = 1,
        val_max_samples: int = 100000,
        # --- Added Warmup Params ---
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        # --- Visualization Params ---
        visualization_method: str = "tsne", # "umap" or "tsne"
        tsne_viz_dir: str = "results/tsne_plots",
        umap_viz_dir: str = "results/umap_plots",
    ):
        """
        Args:
            input_embedding_dim: Dimension of input protein embeddings.
            projection_hidden_dims: List of hidden layer sizes for MLP projection head.
            output_embedding_dim: Dimension of the final contrastive embedding space.
            dropout: Dropout probability in the projection head.
            learning_rate: Optimizer learning rate.
            weight_decay: Optimizer weight decay (L2 regularization).
            use_layer_norm: Whether to use Layer Normalization in the projection head.
            lr_scheduler_config: Config for LR scheduler.
            knn_val_neighbors: Number of neighbors for validation kNN.
            val_max_samples: Maximum validation samples to use for kNN.
            warmup_epochs: Number of epochs for linear LR warmup. 0 disables warmup.
            warmup_start_factor: Initial LR factor (lr = base_lr * factor).
            visualization_method: Method for dimensionality reduction ('umap' or 'tsne').
            tsne_viz_dir: Directory to save t-SNE visualizations.
            umap_viz_dir: Directory to save UMAP visualizations.
        """
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters()
        # Ensure warmup factor is valid
        if not (0 < self.hparams.warmup_start_factor <= 1.0):
             raise ValueError("warmup_start_factor must be > 0 and <= 1.0")
        if self.hparams.warmup_epochs < 0:
             raise ValueError("warmup_epochs cannot be negative.")
        # Validate visualization method
        if self.hparams.visualization_method not in ["umap", "tsne"]:
             raise ValueError(f"Invalid visualization_method: {self.hparams.visualization_method}. Choose 'umap' or 'tsne'.")

        # Build the projection network (MLP)
        self.projection = self._build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm
        )

        # Use the soft triplet loss function
        self.loss_fn = soft_triplet_loss
        # NOTE: `soft_triplet_loss` uses `pairwise_distance_optimized` by default

        # Triplet miner
        self.miner = BatchHardMiner(distance_metric_func=pairwise_distance_optimized)

        # Initialize weights
        self._init_weights()

        # Lists to store validation/test outputs
        self._val_outputs = []
        self._test_outputs = []

        # Create visualization directories
        os.makedirs(self.hparams.tsne_viz_dir, exist_ok=True)
        os.makedirs(self.hparams.umap_viz_dir, exist_ok=True)

    def _build_projection_network(
        self, input_dim: int, hidden_dims: List[int], output_dim: int,
        dropout: float, use_layer_norm: bool
    ) -> nn.Sequential:
        """Helper function to build the MLP projection head."""
        layers: List[nn.Module] = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_dim = h_dim

        # Final projection layer
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming Normal for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Projects input embeddings and applies L2 normalization."""
        projected_embeddings = self.projection(x)
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=1)
        return normalized_embeddings

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Performs a training step with triplet mining."""
        embeddings, labels = batch
        projected_embeddings = self(embeddings)
        
        # Mine hard triplets within the batch
        anchor_idx, positive_idx, negative_idx = self.miner(projected_embeddings, labels)

        active_triplets = len(anchor_idx)
        if active_triplets == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            # Select the embeddings corresponding to the mined triplet indices
            anchor_emb = projected_embeddings[anchor_idx]
            positive_emb = projected_embeddings[positive_idx]
            negative_emb = projected_embeddings[negative_idx]
            loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)

        # Log metrics
        if batch_idx % 100 == 0:
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log('train/active_triplets', float(active_triplets), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    # --- Validation ---
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Stores projected validation embeddings for epoch-end evaluation."""
        embeddings, labels = batch
        with torch.inference_mode():
            projected_embeddings = self(embeddings)

        self._val_outputs.append({
            "embeddings": projected_embeddings.detach(),
            "labels": labels.detach()
        })

    def _shared_epoch_end(self, outputs: List[Dict[str, Tensor]], stage: str):
        """Common logic for validation_epoch_end and test_epoch_end."""
        if not outputs:
            log.warning(f"No outputs collected for stage '{stage}'. Skipping metrics calculation.") # Added warning
            self.log(f"{stage}/knn_acc", 0.0, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/knn_balanced_acc", 0.0, prog_bar=True, sync_dist=True)
            return {} # Return empty dict for consistency

        # Initialize variables that might be deleted in finally block
        all_embeddings = None
        all_labels = None
        dist_matrix = None
        intra_dists_tensor = None
        inter_dists_tensor = None
        metrics = {} # Dictionary to store computed metrics

        try:
            # Concatenate embeddings and labels (currently on CPU for test stage)
            all_embeddings = torch.cat([x["embeddings"] for x in outputs])
            all_labels = torch.cat([x["labels"] for x in outputs])

            # --- Move concatenated tensors to the primary device ---
            all_embeddings = all_embeddings.to(self.device)
            all_labels = all_labels.to(self.device)
            log.debug(f"Moved concatenated tensors to {self.device} for stage '{stage}'.")

            # --- Optional Visualization (only during validation, not testing) ---
            if stage == "val":
                # Generate visualization every 10 epochs, but *only* if not in the sanity checking phase.
                # And only if the trainer is available (might not be during pure inference)
                if hasattr(self, 'trainer') and not self.trainer.sanity_checking and self.current_epoch % 10 == 0:
                    log.info(f"Generating {self.hparams.visualization_method} plot for epoch {self.current_epoch}, stage {stage}...")
                    if self.hparams.visualization_method == "umap":
                        generate_umap_plot(self, all_embeddings, all_labels) # Call UMAP function
                    elif self.hparams.visualization_method == "tsne":
                        generate_tsne_plot(self, all_embeddings, all_labels) # Call t-SNE function
                elif stage == "val" and self.current_epoch % 10 != 0:
                     log.debug(f"Skipping visualization for epoch {self.current_epoch} (not a multiple of 10).")


            # Sample data if too large (use val_max_samples for both stages for simplicity)
            num_samples = all_embeddings.size(0)
            max_samples = self.hparams.val_max_samples # Reuse validation setting
            if num_samples > max_samples:
                log.info(f"Sampling {max_samples} / {num_samples} embeddings for {stage} metrics calculation.")
                indices = torch.randperm(num_samples, device=self.device)[:max_samples]
                all_embeddings = all_embeddings[indices]
                all_labels = all_labels[indices]
            else:
                log.info(f"Using all {num_samples} embeddings for {stage} metrics calculation.")


            # Compute pairwise distances
            # Ensure embeddings are on the correct device before distance calculation
            dist_matrix = pairwise_distance_optimized(all_embeddings, all_embeddings)
            dist_matrix.fill_diagonal_(float('inf'))  # Exclude self

            # --- Add distance statistics ---
            # Calculate intra-class and inter-class distances
            unique_labels = torch.unique(all_labels)
            intra_dists = []
            inter_dists = []

            # Sample subset for efficiency if needed
            max_pairs_per_class = 1000  # Limit computation for large datasets

            log.debug(f"Calculating distance metrics for {len(unique_labels)} unique labels in stage '{stage}'...")
            for label_idx, label in enumerate(unique_labels):
                # Ensure masks use labels on the same device as embeddings
                mask_same = all_labels == label
                mask_diff = all_labels != label

                embeddings_same = all_embeddings[mask_same]
                num_same = embeddings_same.size(0)

                # Sample if too large
                if num_same > max_pairs_per_class:
                    idx = torch.randperm(num_same)[:max_pairs_per_class]
                    embeddings_same_sampled = embeddings_same[idx]
                else:
                    embeddings_same_sampled = embeddings_same

                # Compute intra-class distances
                if num_same > 1:
                    # Ensure embeddings are on the correct device
                    dist_same = pairwise_distance_optimized(embeddings_same_sampled.to(self.device), embeddings_same_sampled.to(self.device))
                    # Create eye mask on the correct device
                    eye_mask = torch.eye(embeddings_same_sampled.size(0), dtype=torch.bool, device=dist_same.device)
                    # Filter out diagonal and collect valid distances
                    valid_dist_same = dist_same[~eye_mask]
                    intra_dists.append(valid_dist_same[torch.isfinite(valid_dist_same)])


                # Sample inter-class
                num_diff = torch.sum(mask_diff).item()
                if num_diff > 0 and num_same > 0:
                    embeddings_diff = all_embeddings[mask_diff]

                    if embeddings_diff.size(0) > max_pairs_per_class:
                        idx = torch.randperm(embeddings_diff.size(0))[:max_pairs_per_class]
                        embeddings_diff_sampled = embeddings_diff[idx]
                    else:
                         embeddings_diff_sampled = embeddings_diff

                    # Compute inter-class distances
                    # Ensure embeddings are on the correct device
                    dist_diff = pairwise_distance_optimized(embeddings_same_sampled.to(self.device), embeddings_diff_sampled.to(self.device))
                    inter_dists.append(dist_diff[torch.isfinite(dist_diff)].flatten()) # Flatten potential matrix

                # Log progress periodically if many labels
                if (label_idx + 1) % 100 == 0:
                     log.debug(f"  Processed distances for {label_idx + 1}/{len(unique_labels)} labels...")


            log.debug("Finished calculating raw distances. Aggregating stats...")
            # Calculate statistics if we have data
            if intra_dists and inter_dists:
                # Concatenate non-empty tensors
                valid_intra = [d for d in intra_dists if d.numel() > 0]
                valid_inter = [d for d in inter_dists if d.numel() > 0]

                if valid_intra and valid_inter:
                    intra_dists_tensor = torch.cat(valid_intra)
                    inter_dists_tensor = torch.cat(valid_inter)

                    metrics[f"{stage}/mean_intra_dist"] = intra_dists_tensor.mean().item()
                    metrics[f"{stage}/mean_inter_dist"] = inter_dists_tensor.mean().item()
                    metrics[f"{stage}/min_inter_dist"] = inter_dists_tensor.min().item()
                    metrics[f"{stage}/max_intra_dist"] = intra_dists_tensor.max().item()

                    # Discriminability ratio (higher is better)
                    if metrics[f"{stage}/mean_intra_dist"] > 1e-6: # Avoid division by zero
                        metrics[f"{stage}/inter_intra_ratio"] = metrics[f"{stage}/mean_inter_dist"] / metrics[f"{stage}/mean_intra_dist"]

                    # Distance margin (higher is better)
                    metrics[f"{stage}/dist_margin"] = metrics[f"{stage}/mean_inter_dist"] - metrics[f"{stage}/mean_intra_dist"]

                    # Overlap measure: percentage of intra-distances larger than smallest inter-distance
                    overlap = torch.mean((intra_dists_tensor > metrics[f"{stage}/min_inter_dist"]).float()).item()
                    metrics[f"{stage}/class_overlap"] = overlap

                    log.debug("Calculated basic distance stats.")

                    # Embedding space quality metrics (computed on subset for efficiency)
                    if all_embeddings.size(0) > 100:
                         log.debug("Calculating embedding uniformity...")
                         # Sample random subset
                         subset_size = min(1000, all_embeddings.size(0))
                         subset_idx = torch.randperm(all_embeddings.size(0))[:subset_size]
                         emb_subset = all_embeddings[subset_idx].cpu() # Move to CPU for pdist

                         # 1. Embedding uniformity
                         with torch.no_grad():
                              pdist_sq = torch.pdist(emb_subset).pow(2)
                              # Add epsilon to avoid log(0) if distances are identical
                              uniformity = pdist_sq.mul(-2).exp().mean().add(1e-8).log().item()
                         metrics[f"{stage}/embedding_uniformity"] = uniformity
                         log.debug(f"  Uniformity: {uniformity:.4f}")


                    # 2. Triplet quality metrics (Adjusted for soft margin)
                    if intra_dists_tensor.numel() > 0 and inter_dists_tensor.numel() > 0:
                        log.debug("Calculating triplet violation rate...")
                        common_sample_size = min(10000, intra_dists_tensor.numel(), inter_dists_tensor.numel())
                        intra_sample = intra_dists_tensor[torch.randperm(intra_dists_tensor.numel())[:common_sample_size]]
                        inter_sample = inter_dists_tensor[torch.randperm(inter_dists_tensor.numel())[:common_sample_size]]

                        # Use broadcasting to compare all pairs between samples
                        # Move samples to the same device for comparison
                        violation_rate = torch.mean(
                            (intra_sample.unsqueeze(1) > inter_sample.unsqueeze(0)).float()
                        ).item()
                        metrics[f"{stage}/triplet_violation_rate"] = violation_rate
                        log.debug(f"  Violation Rate: {violation_rate:.4f}")

                else:
                    log.warning(f"Could not compute detailed distance metrics for stage '{stage}' due to missing valid intra or inter distances.")
            else:
                 log.warning(f"Could not compute any distance metrics for stage '{stage}' (no intra/inter distances found).")


            # Get nearest neighbor
            log.debug("Calculating k-NN metrics...")
            k = self.hparams.knn_val_neighbors # Use same k for test
            # Ensure dist_matrix is on self.device, indices will also be on self.device
            _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

            # Ensure neighbor_labels are computed correctly based on device of all_labels
            # -> Both all_labels and indices are on self.device
            neighbor_labels = all_labels[indices] # Fetch labels using GPU indices

            # For k=1, just use the nearest neighbor's label
            if k == 1:
                predicted_labels = neighbor_labels.squeeze(1)
            else:
                # Majority voting for k>1
                predicted_labels, _ = torch.mode(neighbor_labels, dim=1)

            # Move to CPU for sklearn metrics
            y_true = all_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            # Store metrics
            metrics[f"{stage}/knn_acc"] = accuracy
            metrics[f"{stage}/knn_balanced_acc"] = balanced_accuracy
            log.info(f"Stage '{stage}' Metrics: kNN Acc={accuracy:.4f}, Balanced Acc={balanced_accuracy:.4f}")


        except Exception as e:
            log.error(f"Error calculating metrics for stage '{stage}': {e}", exc_info=True) # Log traceback
            # Log default values if error occurs
            metrics[f"{stage}/knn_acc"] = 0.0
            metrics[f"{stage}/knn_balanced_acc"] = 0.0
            # Log other potential metrics as 0 too, or handle as needed
            metrics[f"{stage}/mean_intra_dist"] = 0.0
            metrics[f"{stage}/mean_inter_dist"] = 0.0
            # ... add others if needed


        finally:
            # Explicitly delete large tensors, checking if they exist first
            if all_embeddings is not None:
                 del all_embeddings
            if all_labels is not None:
                 del all_labels
            if dist_matrix is not None:
                 del dist_matrix
            if intra_dists_tensor is not None:
                 del intra_dists_tensor
            if inter_dists_tensor is not None:
                 del inter_dists_tensor
            # Clear outputs again just in case an error happened before the clear() in try block
            outputs.clear()


        return metrics # Return the computed metrics

    def on_validation_epoch_end(self) -> None:
        """Computes k-NN validation metrics and generates visualization every 10 epochs based on config."""
        # Simply call the shared logic
        metrics = self._shared_epoch_end(self._val_outputs, "val")
        # self._val_outputs is cleared within _shared_epoch_end

        # Explicitly log the computed metrics for callbacks
        if metrics: # Ensure metrics dictionary is not empty
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    # --- Testing ---
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Stores projected test embeddings for epoch-end evaluation."""
        embeddings, labels = batch
        with torch.inference_mode():
            # Project on the current device (GPU)
            projected_embeddings = self(embeddings) # self() handles moving input batch if needed

        # Append to the dedicated test outputs list, moving to CPU for storage
        self._test_outputs.append({
            "embeddings": projected_embeddings.detach().cpu(),
            "labels": labels.detach().cpu() # Assuming labels were already on CPU or need moving
        })

    def on_test_epoch_end(self) -> None:
        """Computes k-NN test metrics."""
        # Call the shared logic with test outputs and stage
        metrics = self._shared_epoch_end(self._test_outputs, "test")
        # self._test_outputs is cleared within _shared_epoch_end

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate, # Base LR
            weight_decay=self.hparams.weight_decay
        )

        if not self.hparams.lr_scheduler_config:
            log.info("No LR scheduler configured.")
            return optimizer

        # Configure the main scheduler (ReduceLROnPlateau)
        # Warmup will be handled in lr_scheduler_step
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler_config.get("mode", "max"),
            factor=self.hparams.lr_scheduler_config.get("factor", 0.5),
            patience=self.hparams.lr_scheduler_config.get("patience", 5),
            min_lr=self.hparams.lr_scheduler_config.get("min_lr", 1e-7)
        )
        log.info(f"Configured ReduceLROnPlateau scheduler with monitor '{self.hparams.lr_scheduler_config.get('monitor')}'")
        if self.hparams.warmup_epochs > 0:
             log.info(f"Using linear LR warmup for {self.hparams.warmup_epochs} epochs, starting from factor {self.hparams.warmup_start_factor:.3f}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.lr_scheduler_config.get("monitor", "val/knn_balanced_acc"),
                "interval": "epoch",
                "frequency": 1
                # No 'strict' needed here as ReduceLROnPlateau uses the monitor
            }
        }

    def lr_scheduler_step(self, scheduler, metric: Optional[Any]):
        """
        Manually handles LR scheduling step to implement linear warmup
        followed by ReduceLROnPlateau.
        """
        # Get optimizer from trainer
        optimizer = self.trainer.optimizers[0]
        current_epoch = self.current_epoch
        warmup_epochs = self.hparams.warmup_epochs
        base_lr = self.hparams.learning_rate
        start_factor = self.hparams.warmup_start_factor

        # --- Warmup Phase ---
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            # Calculate linear warmup scale
            # epoch 0 -> start_factor + (1 - start_factor) * 1 / W
            # epoch W-1 -> start_factor + (1 - start_factor) * W / W = 1.0
            lr_scale = start_factor + (1.0 - start_factor) * (current_epoch + 1) / warmup_epochs
            current_lr = base_lr * lr_scale

            # Manually set learning rate for all parameter groups
            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            # We don't step the ReduceLROnPlateau scheduler during warmup
            # log.debug(f"Warmup Epoch {current_epoch + 1}/{warmup_epochs}: Set LR to {current_lr:.2e}")

        # --- Post-Warmup Phase ---
        else:
            if current_epoch == warmup_epochs and warmup_epochs > 0:
                 # Ensure base LR is set exactly at the end of warmup
                 for pg in optimizer.param_groups:
                     pg['lr'] = base_lr
                 log.info(f"Warmup complete. Set LR to base {base_lr:.2e}")

            # Step the ReduceLROnPlateau scheduler using the monitored metric
            if isinstance(scheduler, ReduceLROnPlateau):
                if metric is None:
                    # This can happen if validation is skipped or the metric isn't available
                    log.warning(f"Epoch {current_epoch}: Metric for ReduceLROnPlateau is None. Skipping scheduler step.")
                else:
                    scheduler.step(metric)
                    # ReduceLROnPlateau doesn't change LR immediately, LearningRateMonitor will log it
                    # log.debug(f"Epoch {current_epoch}: Stepped ReduceLROnPlateau with metric {metric:.4f}")
            else:
                # Fallback for other schedulers if configured differently in the future
                scheduler.step()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Generates embeddings for prediction."""
        if isinstance(batch, tuple) and len(batch) > 0:
            embeddings = batch[0].to(self.device)
        elif isinstance(batch, torch.Tensor):
            embeddings = batch.to(self.device)
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")
            
        with torch.inference_mode():
            return self(embeddings).cpu()