import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor # Explicit type hinting
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import seaborn as sns
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau # Explicit import
from sklearn.decomposition import PCA

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
    def __init__(self, distance_metric_func=pairwise_distance_optimized, margin=0.5):
        self.distance_metric = distance_metric_func
        self.margin = margin

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

        # --- Find Semi-Hard Negatives Efficiently ---
        # Get distance threshold matrix for each anchor (based on its hardest positive)
        # This is an efficient replacement for the loop
        # Broadcasting hardest_pos_dist to create thresholds for each anchor
        pos_dist_threshold = hardest_pos_dist.unsqueeze(1).expand(-1, batch_size)
        
        # Create masks for semi-hard negatives (between pos_dist and pos_dist + margin)
        semi_hard_mask = (dist_mat > pos_dist_threshold) & (dist_mat < pos_dist_threshold + self.margin)
        
        # Only consider actual negatives (different class)
        semi_hard_mask = semi_hard_mask & labels_not_equal
        
        # Don't consider self as negative
        semi_hard_mask = semi_hard_mask & ~identity_mask
        
        # Create negative distance matrix with only semi-hard negatives
        neg_dist_mat = dist_mat.clone()
        neg_dist_mat.masked_fill_(~semi_hard_mask, torch.inf)
        
        # If no semi-hard negatives exist for an anchor, fall back to the hardest negatives
        no_semi_hard = (neg_dist_mat == torch.inf).all(dim=1)
        if no_semi_hard.any():
            hard_neg_dist_mat = dist_mat.clone()
            hard_neg_dist_mat.masked_fill_(~labels_not_equal | identity_mask, torch.inf)
            # Only replace rows that don't have semi-hard negatives
            neg_dist_mat[no_semi_hard] = hard_neg_dist_mat[no_semi_hard]
        
        # Get minimum distance negative (closest semi-hard or hard negative)
        hardest_neg_dist, hardest_neg_idx = torch.min(neg_dist_mat, dim=1)

        # --- Filter Valid Triplets ---
        valid_pos_mask = hardest_pos_dist > -torch.inf
        valid_neg_mask = hardest_neg_dist < torch.inf
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
        triplet_margin: float = 0.5,
        use_layer_norm: bool = True,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        knn_val_neighbors: int = 1,
        val_max_samples: int = 100000,
        # --- Added Warmup Params ---
        warmup_epochs: int = 0,
        warmup_start_factor: float = 0.1,
        # Add simple viz option
        tsne_viz_dir: str = "results/tsne_plots",
    ):
        """
        Args:
            input_embedding_dim: Dimension of input protein embeddings.
            projection_hidden_dims: List of hidden layer sizes for MLP projection head.
            output_embedding_dim: Dimension of the final contrastive embedding space.
            dropout: Dropout probability in the projection head.
            learning_rate: Optimizer learning rate.
            weight_decay: Optimizer weight decay (L2 regularization).
            triplet_margin: Margin for the TripletMarginLoss.
            use_layer_norm: Whether to use Layer Normalization in the projection head.
            lr_scheduler_config: Config for LR scheduler.
            knn_val_neighbors: Number of neighbors for validation kNN.
            val_max_samples: Maximum validation samples to use for kNN.
            warmup_epochs: Number of epochs for linear LR warmup. 0 disables warmup.
            warmup_start_factor: Initial LR factor (lr = base_lr * factor).
            tsne_viz_dir: Directory to save t-SNE visualizations
        """
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters()
        # Ensure warmup factor is valid
        if not (0 < self.hparams.warmup_start_factor <= 1.0):
             raise ValueError("warmup_start_factor must be > 0 and <= 1.0")
        if self.hparams.warmup_epochs < 0:
             raise ValueError("warmup_epochs cannot be negative.")

        # Build the projection network (MLP)
        self.projection = self._build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm
        )

        # Loss function with squared Euclidean distance
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=pairwise_distance_optimized,
            margin=self.hparams.triplet_margin,
            reduction='mean'
        )

        # Triplet miner
        self.miner = BatchHardMiner(distance_metric_func=pairwise_distance_optimized)

        # Initialize weights
        self._init_weights()

        # Lists to store validation/test outputs
        self._val_outputs = []
        self._test_outputs = []

        # Create visualization directory
        os.makedirs(self.hparams.tsne_viz_dir, exist_ok=True)

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

    def on_validation_epoch_end(self) -> None:
        """Computes k-NN validation metrics and generates simple t-SNE visualization every 10 epochs."""
        if not self._val_outputs:
            self.log("val/knn_acc", 0.0, prog_bar=True, sync_dist=True)
            self.log("val/knn_balanced_acc", 0.0, prog_bar=True, sync_dist=True)
            return

        try:
            # Concatenate embeddings and labels
            all_embeddings = torch.cat([x["embeddings"] for x in self._val_outputs])
            all_labels = torch.cat([x["labels"] for x in self._val_outputs])
            
            # Generate t-SNE visualization every 10 epochs, but *only* if not in the sanity checking phase.
            if not self.trainer.sanity_checking and self.current_epoch % 10 == 0:
                self._generate_tsne_plot(all_embeddings, all_labels)
            
            self._val_outputs.clear()  # Free memory

            # Sample validation data if too large
            num_samples = all_embeddings.size(0)
            if num_samples > self.hparams.val_max_samples:
                indices = torch.randperm(num_samples, device=self.device)[:self.hparams.val_max_samples]
                all_embeddings = all_embeddings[indices]
                all_labels = all_labels[indices]

            # Compute pairwise distances
            dist_matrix = pairwise_distance_optimized(all_embeddings, all_embeddings)
            dist_matrix.fill_diagonal_(float('inf'))  # Exclude self

            # --- Add distance statistics ---
            # Calculate intra-class and inter-class distances
            unique_labels = torch.unique(all_labels)
            intra_dists = []
            inter_dists = []
            
            # Sample subset for efficiency if needed
            max_pairs_per_class = 1000  # Limit computation for large datasets
            
            for label in unique_labels:
                mask_same = all_labels == label
                mask_diff = all_labels != label
                
                # Get embeddings for this class
                embeddings_same = all_embeddings[mask_same]
                
                # Sample if too large
                if embeddings_same.size(0) > max_pairs_per_class:
                    idx = torch.randperm(embeddings_same.size(0))[:max_pairs_per_class]
                    embeddings_same = embeddings_same[idx]
                    
                # Compute intra-class distances (distances within the same class)
                if embeddings_same.size(0) > 1:
                    dist_same = pairwise_distance_optimized(embeddings_same, embeddings_same)
                    dist_same = dist_same[~torch.eye(embeddings_same.size(0), dtype=bool, device=dist_same.device)]
                    intra_dists.append(dist_same)
                
                # Sample inter-class
                if torch.sum(mask_diff) > 0 and embeddings_same.size(0) > 0:
                    embeddings_diff = all_embeddings[mask_diff]
                    if embeddings_diff.size(0) > max_pairs_per_class:
                        idx = torch.randperm(embeddings_diff.size(0))[:max_pairs_per_class]
                        embeddings_diff = embeddings_diff[idx]
                    
                    # Compute inter-class distances (distances between different classes)
                    dist_diff = pairwise_distance_optimized(embeddings_same, embeddings_diff)
                    inter_dists.append(dist_diff)
            
            # Calculate statistics if we have data
            if intra_dists and inter_dists:
                intra_dists = torch.cat(intra_dists)
                inter_dists = torch.cat(inter_dists)
                
                mean_intra = intra_dists.mean().item()
                mean_inter = inter_dists.mean().item()
                min_inter = inter_dists.min().item()
                max_intra = intra_dists.max().item()
                
                # Log distance metrics
                self.log("val/mean_intra_dist", mean_intra, sync_dist=True)
                self.log("val/mean_inter_dist", mean_inter, sync_dist=True)
                self.log("val/min_inter_dist", min_inter, sync_dist=True)
                self.log("val/max_intra_dist", max_intra, sync_dist=True)
                
                # Discriminability ratio (higher is better)
                if mean_intra > 0:
                    self.log("val/inter_intra_ratio", mean_inter / mean_intra, sync_dist=True)
                
                # Distance margin (higher is better)
                self.log("val/dist_margin", mean_inter - mean_intra, sync_dist=True)
                self.log("val/margin_buffer", (mean_inter - mean_intra - self.hparams.triplet_margin), sync_dist=True)
                
                # Overlap measure: percentage of intra-distances larger than smallest inter-distance
                overlap = torch.mean((intra_dists > min_inter).float()).item()
                self.log("val/class_overlap", overlap, sync_dist=True)


                
                # Embedding space quality metrics (computed on subset for efficiency)
                if all_embeddings.size(0) > 100:
                    # Sample random subset
                    subset_size = min(1000, all_embeddings.size(0))
                    subset_idx = torch.randperm(all_embeddings.size(0))[:subset_size]
                    emb_subset = all_embeddings[subset_idx].cpu()
                    
                    # 1. Embedding uniformity (measures how uniform the distribution is on unit sphere)
                    # Lower is better - points more uniformly distributed
                    with torch.no_grad():
                        uniformity = torch.pdist(emb_subset).pow(2).mul(-2).exp().mean().log().item()
                    self.log("val/embedding_uniformity", uniformity, sync_dist=True)
                    
                # 2. Triplet quality metrics
                if len(intra_dists) > 0 and len(inter_dists) > 0:
                    # Triplet violation rate: % of potential triplets violating margin constraint
                    # (intra_dist > inter_dist - margin)
                    margin = self.hparams.triplet_margin
                    # Use the same sample size for both to avoid dimension mismatch
                    common_sample_size = min(10000, len(intra_dists), len(inter_dists))
                    intra_sample = intra_dists[torch.randperm(len(intra_dists))[:common_sample_size]]
                    inter_sample = inter_dists[torch.randperm(len(inter_dists))[:common_sample_size]]
                    violation_rate = torch.mean(
                        (intra_sample.unsqueeze(1) > inter_sample.unsqueeze(0) - margin).float()
                    ).item()
                    self.log("val/triplet_violation_rate", violation_rate, sync_dist=True)

            # Get nearest neighbor
            k = self.hparams.knn_val_neighbors
            _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)
            neighbor_labels = all_labels[indices]

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

            # Log metrics
            self.log("val/knn_acc", accuracy, prog_bar=True, sync_dist=True)
            self.log("val/knn_balanced_acc", balanced_accuracy, prog_bar=True, sync_dist=True)
            
        except Exception as e:
            log.error(f"Error in validation metrics: {e}")
            self.log("val/knn_acc", 0.0, prog_bar=True, sync_dist=True)
            self.log("val/knn_balanced_acc", 0.0, prog_bar=True, sync_dist=True)

    def _generate_tsne_plot(self, embeddings: Tensor, labels: Tensor) -> None:
        """
        Creates a minimalist visualization of embeddings using PCA preprocessing
        followed by t-SNE, colored by CATH classes, following Tufte principles.
        
        Args:
            embeddings: Projected embeddings tensor
            labels: Corresponding label tensor (CATH class IDs)
        """
        try:
            # Start timing
            start_time = time.time()
            log.info(f"Generating PCA+t-SNE plot for epoch {self.current_epoch}")
            
            # Sample for efficiency (max 10000 points)
            max_samples = 10000
            if len(embeddings) > max_samples:
                indices = torch.randperm(len(embeddings))[:max_samples]
                embeddings_subset = embeddings[indices].cpu().numpy()
                labels_subset = labels[indices].cpu().numpy()
            else:
                embeddings_subset = embeddings.cpu().numpy()
                labels_subset = labels.cpu().numpy()
            
            # Get unique labels
            unique_labels = np.unique(labels_subset)
            
            cath_class_names = {
                0: "Mainly Alpha",
                1: "Mainly Beta",
                2: "Alpha Beta", 
                3: "Few Secondary Structures"
            }
            
            # Simple PCA preprocessing
            n_components = min(50, embeddings_subset.shape[1])
            pca = PCA(n_components=n_components, random_state=42)
            embeddings_pca = pca.fit_transform(embeddings_subset)
            
            # t-SNE on PCA results
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_result = tsne.fit_transform(embeddings_pca)
            
            # Set up minimalist plot with Tufte-inspired style
            plt.figure(figsize=(8, 6))
            
            # Set clean style
            plt.rcParams['axes.spines.top'] = False
            plt.rcParams['axes.spines.right'] = False
            plt.rcParams['axes.grid'] = False
            
            # Create color mapping that maintains consistent colors per CATH class
            color_palette = sns.color_palette("colorblind", n_colors=len(unique_labels))
            colors = [color_palette[int(label)] for label in labels_subset]
            
            # Create scatter plot - smaller points with higher density
            plt.scatter(
                tsne_result[:, 0], tsne_result[:, 1],
                c=colors,
                s=5,  # Smaller point size
                alpha=0.7,  # Slightly transparent
                linewidths=0,  # No edge lines
                rasterized=True  # Better for export
            )
            
            # Minimal labels and subtle tick marks
            plt.title(f'CATH Classes (Epoch {self.current_epoch})', fontsize=12, pad=10)
            plt.tick_params(axis='both', which='major', labelsize=8, length=3, width=0.5)
            
            # Subtle axis labels
            plt.xlabel('t-SNE Dimension 1', fontsize=9, labelpad=7, color='#505050')
            plt.ylabel('t-SNE Dimension 2', fontsize=9, labelpad=7, color='#505050')
            
            # Create minimal legend with CATH class names
            if len(unique_labels) <= 4:
                handles = [plt.Line2D([0], [0], marker='o', color=color_palette[i], 
                                      markersize=5, linestyle='',
                                      label=cath_class_names.get(unique_labels[i], f"Class {unique_labels[i]}"))
                           for i in range(len(unique_labels))]
                plt.legend(handles=handles,
                          loc='best',
                          frameon=False,  # No frame
                          fontsize=9,
                          handletextpad=0.5)  # Less space between marker and text
            
            # Tighter layout with reduced margins
            plt.tight_layout(pad=1.2)
            
            # Save the plot
            filename = f"tsne_epoch_{self.current_epoch}.png"
            save_path = os.path.join(self.hparams.tsne_viz_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
            plt.close()
            
            # Log completion
            elapsed_time = time.time() - start_time
            log.info(f"t-SNE plot saved to {save_path} (took {elapsed_time:.2f}s)")
            
        except Exception as e:
            log.error(f"Error generating t-SNE plot: {e}")
            log.exception("Detailed traceback:")

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
            min_lr=self.hparams.lr_scheduler_config.get("min_lr", 1e-7),
            verbose=False # We will log LR manually or rely on LearningRateMonitor
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