import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor
import os
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler


from plotting import generate_tsne_plot, generate_umap_plot
from distances import pairwise_distance_optimized
from losses import soft_triplet_loss
from mining import BatchHardMiner, SemiHardMiner

# Configure logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
)

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
        # --- Miner Selection ---
        miner_type: str = "batch_hard", # "batch_hard" or "semi_hard"
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
            miner_type: Which triplet mining strategy to use ('batch_hard' or 'semi_hard').
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
        # Validate miner type
        if self.hparams.miner_type not in ["batch_hard", "semi_hard"]:
            raise ValueError(f"Invalid miner_type: {self.hparams.miner_type}. Choose 'batch_hard' or 'semi_hard'.")

        # Build the projection network (MLP)
        self.projection = self._build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm
        )

        # Use the imported soft triplet loss function
        self.loss_fn = soft_triplet_loss
        # NOTE: `soft_triplet_loss` uses `pairwise_distance_optimized` by default

        # Select and initialize the triplet miner based on config
        distance_func = pairwise_distance_optimized # Use imported function
        if self.hparams.miner_type == "batch_hard":
            self.miner = BatchHardMiner(distance_metric_func=distance_func)
            log.info("Using Batch Hard Miner.")
        elif self.hparams.miner_type == "semi_hard":
            self.miner = SemiHardMiner(distance_metric_func=distance_func)
            log.info("Using Semi Hard Miner.")

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
        projected_embeddings = self(embeddings) # Apply projection and normalization

        # Mine hard triplets within the batch using the selected miner
        anchor_idx, positive_idx, negative_idx = self.miner(projected_embeddings, labels)

        active_triplets = len(anchor_idx)
        if active_triplets == 0:
            # Handle case with no valid triplets to avoid errors
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            # Potentially log or track this occurrence if it happens frequently
            # log.debug(f"Batch {batch_idx}: No active triplets found.")
        else:
            # Select the embeddings corresponding to the mined triplet indices
            anchor_emb = projected_embeddings[anchor_idx]
            positive_emb = projected_embeddings[positive_idx]
            negative_emb = projected_embeddings[negative_idx]

            # Calculate loss using the imported loss function
            loss = self.loss_fn(anchor_emb, positive_emb, negative_emb) # Pass the correct distance func if needed

        # Log metrics
        if batch_idx % 100 == 0: # Log less frequently on step
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log('train/active_triplets', float(active_triplets), on_step=True, on_epoch=True, prog_bar=False, sync_dist=True) # Log on step too

        return loss

    # --- Validation ---
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Stores projected validation embeddings for epoch-end evaluation."""
        embeddings, labels = batch
        with torch.inference_mode():
            projected_embeddings = self(embeddings)

        self._val_outputs.append({
            "embeddings": projected_embeddings.detach(), # Keep on device for epoch end
            "labels": labels.detach() # Keep on device for epoch end
        })

    def _generate_visualization_if_needed(
        self, all_embeddings: Tensor, all_labels: Tensor, stage: str
    ) -> None:
        """Generates dimensionality reduction plot during validation if conditions are met."""
        if stage == "val":
            # Generate visualization every 10 epochs, but *only* if not in the sanity checking phase.
            # And only if the trainer is available (might not be during pure inference)
            should_visualize = (
                hasattr(self, 'trainer')
                and not self.trainer.sanity_checking
                and self.current_epoch > 0 # Avoid plotting before first real epoch
                and self.current_epoch % 10 == 0
            )
            if should_visualize:
                log.info(f"Generating {self.hparams.visualization_method} plot for epoch {self.current_epoch}, stage {stage}...")
                # Pass the device embeddings to the plot functions
                if self.hparams.visualization_method == "umap":
                    generate_umap_plot(self, all_embeddings, all_labels) # Call UMAP function
                elif self.hparams.visualization_method == "tsne":
                    generate_tsne_plot(self, all_embeddings, all_labels) # Call t-SNE function
            elif stage == "val" and self.current_epoch % 10 != 0:
                 log.debug(f"Skipping visualization for epoch {self.current_epoch} (not a multiple of 10).")

    def _compute_distance_metrics(
        self, all_embeddings: Tensor, all_labels: Tensor, stage: str
    ) -> Dict[str, float]:
        """Computes various distance-based metrics for the embedding space."""
        metrics = {}
        # Define constants for sampling/subset sizes within this scope
        _MAX_PAIRS_PER_CLASS_DIST = 1000
        _UNIFORMITY_SUBSET_SIZE = 1000
        _TRIPLET_VIOLATION_SAMPLE_SIZE = 10000

        try:
            unique_labels = torch.unique(all_labels)
            intra_dists = []
            inter_dists = []
            current_device = all_embeddings.device # Get device from tensor

            log.debug(f"Calculating distance metrics for {len(unique_labels)} unique labels in stage '{stage}'...")
            for label_idx, label in enumerate(unique_labels):
                mask_same = all_labels == label
                mask_diff = all_labels != label

                embeddings_same = all_embeddings[mask_same]
                num_same = embeddings_same.size(0)

                # Sample if too large
                if num_same > _MAX_PAIRS_PER_CLASS_DIST:
                    idx = torch.randperm(num_same, device=current_device)[:_MAX_PAIRS_PER_CLASS_DIST]
                    embeddings_same_sampled = embeddings_same[idx]
                else:
                    embeddings_same_sampled = embeddings_same

                # Compute intra-class distances
                if num_same > 1:
                    dist_same = pairwise_distance_optimized(embeddings_same_sampled, embeddings_same_sampled)
                    eye_mask = torch.eye(embeddings_same_sampled.size(0), dtype=torch.bool, device=dist_same.device)
                    valid_dist_same = dist_same[~eye_mask]
                    intra_dists.append(valid_dist_same[torch.isfinite(valid_dist_same)])

                # Sample inter-class
                num_diff = torch.sum(mask_diff).item()
                if num_diff > 0 and num_same > 0:
                    embeddings_diff = all_embeddings[mask_diff]

                    if embeddings_diff.size(0) > _MAX_PAIRS_PER_CLASS_DIST:
                        idx = torch.randperm(embeddings_diff.size(0), device=current_device)[:_MAX_PAIRS_PER_CLASS_DIST]
                        embeddings_diff_sampled = embeddings_diff[idx]
                    else:
                         embeddings_diff_sampled = embeddings_diff

                    dist_diff = pairwise_distance_optimized(embeddings_same_sampled, embeddings_diff_sampled)
                    inter_dists.append(dist_diff[torch.isfinite(dist_diff)].flatten())

                if (label_idx + 1) % 100 == 0:
                     log.debug(f"  Processed distances for {label_idx + 1}/{len(unique_labels)} labels...")

            log.debug("Finished calculating raw distances. Aggregating stats...")
            # Calculate statistics if we have data
            if intra_dists and inter_dists:
                valid_intra = [d for d in intra_dists if d.numel() > 0]
                valid_inter = [d for d in inter_dists if d.numel() > 0]

                if valid_intra and valid_inter:
                    intra_dists_tensor = torch.cat(valid_intra)
                    inter_dists_tensor = torch.cat(valid_inter)

                    metrics[f"{stage}/mean_intra_dist"] = intra_dists_tensor.mean().item()
                    metrics[f"{stage}/mean_inter_dist"] = inter_dists_tensor.mean().item()
                    metrics[f"{stage}/min_inter_dist"] = inter_dists_tensor.min().item()
                    metrics[f"{stage}/max_intra_dist"] = intra_dists_tensor.max().item()

                    if metrics[f"{stage}/mean_intra_dist"] > 1e-6:
                        metrics[f"{stage}/inter_intra_ratio"] = metrics[f"{stage}/mean_inter_dist"] / metrics[f"{stage}/mean_intra_dist"]

                    metrics[f"{stage}/dist_margin"] = metrics[f"{stage}/mean_inter_dist"] - metrics[f"{stage}/mean_intra_dist"]

                    overlap = torch.mean((intra_dists_tensor > metrics[f"{stage}/min_inter_dist"]).float()).item()
                    metrics[f"{stage}/class_overlap"] = overlap

                    log.debug("Calculated basic distance stats.")

                    # Embedding space quality metrics
                    if all_embeddings.size(0) > 100: # Ensure enough samples
                         log.debug("Calculating embedding uniformity...")
                         subset_size = min(_UNIFORMITY_SUBSET_SIZE, all_embeddings.size(0))
                         subset_idx = torch.randperm(all_embeddings.size(0), device=current_device)[:subset_size]
                         emb_subset = all_embeddings[subset_idx]

                         try: # GPU Uniformity
                              with torch.no_grad():
                                   pdist_sq = torch.pdist(emb_subset).pow(2)
                                   uniformity = pdist_sq.mul(-2).exp().mean().add(1e-8).log().item()
                                   metrics[f"{stage}/embedding_uniformity"] = uniformity
                                   log.debug(f"  Uniformity: {uniformity:.4f}")
                         except RuntimeError as uniform_err: # Fallback CPU
                              log.warning(f"Could not compute uniformity on device {current_device} ({uniform_err}). Trying on CPU.")
                              try:
                                   emb_subset_cpu = emb_subset.cpu()
                                   with torch.no_grad():
                                        pdist_sq_cpu = torch.pdist(emb_subset_cpu).pow(2)
                                        uniformity_cpu = pdist_sq_cpu.mul(-2).exp().mean().add(1e-8).log().item()
                                   metrics[f"{stage}/embedding_uniformity"] = uniformity_cpu
                                   log.debug(f"  Uniformity (CPU fallback): {uniformity_cpu:.4f}")
                              except Exception as cpu_err:
                                   log.error(f"Failed to compute uniformity on CPU as well: {cpu_err}")
                                   metrics[f"{stage}/embedding_uniformity"] = float('nan')
                    else:
                        metrics[f"{stage}/embedding_uniformity"] = float('nan') # Not enough samples

                    # Triplet quality metrics
                    if intra_dists_tensor.numel() > 0 and inter_dists_tensor.numel() > 0:
                        log.debug("Calculating triplet violation rate...")
                        common_sample_size = min(_TRIPLET_VIOLATION_SAMPLE_SIZE, intra_dists_tensor.numel(), inter_dists_tensor.numel())
                        intra_sample = intra_dists_tensor[torch.randperm(intra_dists_tensor.numel(), device=current_device)[:common_sample_size]]
                        inter_sample = inter_dists_tensor[torch.randperm(inter_dists_tensor.numel(), device=current_device)[:common_sample_size]]

                        violation_rate = torch.mean(
                            (intra_sample.unsqueeze(1) > inter_sample.unsqueeze(0)).float()
                        ).item()
                        metrics[f"{stage}/triplet_violation_rate"] = violation_rate
                        log.debug(f"  Violation Rate: {violation_rate:.4f}")
                    else:
                         metrics[f"{stage}/triplet_violation_rate"] = float('nan')
                else:
                    log.warning(f"Could not compute detailed distance metrics for stage '{stage}' due to missing valid intra or inter distances.")
                    metrics[f"{stage}/mean_intra_dist"] = float('nan')
                    metrics[f"{stage}/mean_inter_dist"] = float('nan')
                    metrics[f"{stage}/embedding_uniformity"] = float('nan')
                    metrics[f"{stage}/triplet_violation_rate"] = float('nan')
            else:
                 log.warning(f"Could not compute any distance metrics for stage '{stage}' (no intra/inter distances found).")
                 metrics[f"{stage}/mean_intra_dist"] = float('nan')
                 metrics[f"{stage}/mean_inter_dist"] = float('nan')
                 metrics[f"{stage}/embedding_uniformity"] = float('nan')
                 metrics[f"{stage}/triplet_violation_rate"] = float('nan')

        except Exception as e:
            log.error(f"Error calculating distance metrics for stage '{stage}': {e}", exc_info=True)
            # Ensure keys exist even on error
            metrics[f"{stage}/mean_intra_dist"] = float('nan')
            metrics[f"{stage}/mean_inter_dist"] = float('nan')
            metrics[f"{stage}/embedding_uniformity"] = float('nan')
            metrics[f"{stage}/triplet_violation_rate"] = float('nan')

        finally:
             # Clean up large intermediate tensors if they were created
            if 'intra_dists_tensor' in locals() and intra_dists_tensor is not None: del intra_dists_tensor
            if 'inter_dists_tensor' in locals() and inter_dists_tensor is not None: del inter_dists_tensor
            if 'intra_dists' in locals(): del intra_dists
            if 'inter_dists' in locals(): del inter_dists

        return metrics

    def _compute_knn_metrics(
        self, dist_matrix: Tensor, all_labels: Tensor, k: int, stage: str
    ) -> Dict[str, float]:
        """Computes k-NN accuracy metrics."""
        metrics = {}
        current_device = dist_matrix.device
        try:
            log.debug(f"Calculating k-NN metrics (k={k})...")
            # Ensure dist_matrix is on self.device, indices will also be on self.device
            _, indices = torch.topk(dist_matrix, k=k, dim=1, largest=False)

            # Fetch labels using GPU indices
            neighbor_labels = all_labels[indices]

            if k == 1:
                predicted_labels = neighbor_labels.squeeze(1)
            else:
                predicted_labels, _ = torch.mode(neighbor_labels, dim=1)

            # Move to CPU for sklearn metrics ONLY at the end
            y_true = all_labels.cpu().numpy()
            y_pred = predicted_labels.cpu().numpy()

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

            metrics[f"{stage}/knn_acc"] = accuracy
            metrics[f"{stage}/knn_balanced_acc"] = balanced_accuracy
            log.info(f"Stage '{stage}' kNN Metrics: Acc={accuracy:.4f}, Balanced Acc={balanced_accuracy:.4f}")

        except Exception as e:
            log.error(f"Error calculating k-NN metrics for stage '{stage}': {e}", exc_info=True)
            metrics[f"{stage}/knn_acc"] = 0.0
            metrics[f"{stage}/knn_balanced_acc"] = 0.0
        finally:
            if 'indices' in locals(): del indices
            if 'neighbor_labels' in locals(): del neighbor_labels
            if 'predicted_labels' in locals(): del predicted_labels

        return metrics


    def _shared_epoch_end(self, outputs: List[Dict[str, Tensor]], stage: str):
        """Common logic for validation_epoch_end and test_epoch_end."""
        if not outputs:
            log.warning(f"No outputs collected for stage '{stage}'. Skipping metrics calculation.")
            self.log(f"{stage}/knn_acc", 0.0, prog_bar=True, sync_dist=True)
            self.log(f"{stage}/knn_balanced_acc", 0.0, prog_bar=True, sync_dist=True)
            outputs.clear() # Clear even if empty
            return {}

        all_embeddings = None
        all_labels = None
        dist_matrix = None
        metrics = {}

        try:
            # Concatenate embeddings and labels
            # Outputs might be on CPU (test) or GPU (val) - handle device transfer
            target_device = self.device
            all_embeddings_list = [x["embeddings"].to(target_device) for x in outputs]
            all_labels_list = [x["labels"].to(target_device) for x in outputs]
            all_embeddings = torch.cat(all_embeddings_list)
            all_labels = torch.cat(all_labels_list)
            log.debug(f"Concatenated tensors moved to {target_device} for stage '{stage}'. Shape: {all_embeddings.shape}")

            # --- Optional Visualization (Validation Only) ---
            self._generate_visualization_if_needed(all_embeddings, all_labels, stage)

            # --- Sampling ---
            num_samples = all_embeddings.size(0)
            max_samples = self.hparams.val_max_samples
            if num_samples > max_samples:
                log.info(f"Sampling {max_samples} / {num_samples} embeddings for {stage} metrics calculation.")
                indices = torch.randperm(num_samples, device=target_device)[:max_samples]
                all_embeddings = all_embeddings[indices]
                all_labels = all_labels[indices]
            else:
                log.info(f"Using all {num_samples} embeddings for {stage} metrics calculation.")

            # --- Compute Pairwise Distances ---
            dist_matrix = pairwise_distance_optimized(all_embeddings, all_embeddings)
            dist_matrix.fill_diagonal_(float('inf')) # Exclude self-distance

            # --- Compute Distance-Based Metrics ---
            distance_metrics = self._compute_distance_metrics(all_embeddings, all_labels, stage)
            metrics.update(distance_metrics)

            # --- Compute k-NN Metrics ---
            knn_metrics = self._compute_knn_metrics(
                dist_matrix, all_labels, self.hparams.knn_val_neighbors, stage
            )
            metrics.update(knn_metrics)

        except Exception as e:
            log.error(f"Error during _shared_epoch_end for stage '{stage}': {e}", exc_info=True)
            # Ensure default keys exist if overall calculation failed early
            metrics.setdefault(f"{stage}/knn_acc", 0.0)
            metrics.setdefault(f"{stage}/knn_balanced_acc", 0.0)
            metrics.setdefault(f"{stage}/mean_intra_dist", float('nan'))
            metrics.setdefault(f"{stage}/mean_inter_dist", float('nan'))
            metrics.setdefault(f"{stage}/embedding_uniformity", float('nan'))
            metrics.setdefault(f"{stage}/triplet_violation_rate", float('nan'))

        finally:
            # Explicitly delete large tensors and clear list
            if all_embeddings is not None: del all_embeddings
            if all_labels is not None: del all_labels
            if dist_matrix is not None: del dist_matrix
            if 'all_embeddings_list' in locals(): del all_embeddings_list
            if 'all_labels_list' in locals(): del all_labels_list
            outputs.clear() # Clear the list passed in

        return metrics # Return the computed metrics

    def on_validation_epoch_end(self) -> None:
        """Computes validation metrics and generates visualization."""
        metrics = self._shared_epoch_end(self._val_outputs, "val")
        # self._val_outputs is cleared within _shared_epoch_end

        if metrics:
            loggable_metrics = {k: v for k, v in metrics.items() if not isinstance(v, float) or not np.isnan(v)}
            self.log_dict(loggable_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
             log.warning("Validation epoch end: No metrics were computed or returned.")


    # --- Testing ---
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """
        Stores projected test embeddings for epoch-end evaluation.
        Outputs are moved to CPU immediately to potentially reduce peak GPU memory
        during the test epoch end, especially for large test sets.
        This adds CPU<->GPU transfer overhead in `_shared_epoch_end` but might
        be necessary for memory-constrained environments.
        """
        embeddings, labels = batch
        with torch.inference_mode():
            projected_embeddings = self(embeddings) # Runs on self.device

        self._test_outputs.append({
            "embeddings": projected_embeddings.detach().cpu(),
            "labels": labels.detach().cpu() # Assuming labels might be on GPU initially
        })

    def on_test_epoch_end(self) -> None:
        """Computes k-NN test metrics."""
        # Test outputs are on CPU, _shared_epoch_end handles moving them back to device.
        metrics = self._shared_epoch_end(self._test_outputs, "test")
        # self._test_outputs is cleared within _shared_epoch_end

        if metrics:
            loggable_metrics = {k: v for k, v in metrics.items() if not isinstance(v, float) or not np.isnan(v)}
            self.log_dict(loggable_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True) # Log test metrics
        else:
             log.warning("Test epoch end: No metrics were computed or returned.")


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
        scheduler_monitor = self.hparams.lr_scheduler_config.get("monitor", "val/knn_balanced_acc") # Default monitor
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler_config.get("mode", "max"),
            factor=self.hparams.lr_scheduler_config.get("factor", 0.5),
            patience=self.hparams.lr_scheduler_config.get("patience", 5),
            min_lr=self.hparams.lr_scheduler_config.get("min_lr", 1e-7),
            verbose=True # Log when LR changes
        )
        log.info(f"Configured ReduceLROnPlateau scheduler with monitor '{scheduler_monitor}'")
        if self.hparams.warmup_epochs > 0:
             log.info(f"Using linear LR warmup for {self.hparams.warmup_epochs} epochs, starting from factor {self.hparams.warmup_start_factor:.3f}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_monitor, # Use the defined monitor value
                "interval": "epoch",
                "frequency": 1
            }
        }

    def lr_scheduler_step(self, scheduler: _LRScheduler, metric: Optional[torch.Tensor] = None):
        """
        Manually handles LR scheduling step to implement linear warmup
        followed by ReduceLROnPlateau.
        """
        if not self.trainer or not self.trainer.optimizers:
            log.warning("lr_scheduler_step called but trainer.optimizers is not available.")
            return # Cannot proceed without optimizer

        optimizer = self.trainer.optimizers[0]
        current_epoch = self.current_epoch
        warmup_epochs = self.hparams.warmup_epochs
        base_lr = self.hparams.learning_rate
        start_factor = self.hparams.warmup_start_factor

        # --- Warmup Phase ---
        if warmup_epochs > 0 and current_epoch < warmup_epochs:
            lr_scale = start_factor + (1.0 - start_factor) * (current_epoch + 1.0) / float(warmup_epochs) # Ensure float division
            current_lr = base_lr * lr_scale

            for pg in optimizer.param_groups:
                pg['lr'] = current_lr

            self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            # log.debug(f"Warmup Epoch {current_epoch + 1}/{warmup_epochs}: Set LR to {current_lr:.2e}")

        # --- Post-Warmup Phase ---
        else:
            if current_epoch == warmup_epochs and warmup_epochs > 0:
                 for pg in optimizer.param_groups:
                     pg['lr'] = base_lr
                 log.info(f"Warmup complete. Set LR to base {base_lr:.2e}")
                 self.log('lr', base_lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True) # Log base LR

            # Step the ReduceLROnPlateau scheduler using the monitored metric
            if isinstance(scheduler, ReduceLROnPlateau):
                if metric is None:
                    if not self.trainer.sanity_checking: # Don't warn during sanity check
                        log.warning(f"Epoch {current_epoch + 1}: Metric for ReduceLROnPlateau is None. Skipping scheduler step.")
                else:
                    scheduler.step(metric) # Step with the metric
                    current_lr = optimizer.param_groups[0]['lr'] # Get potentially updated LR
                    self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
                    # log.debug(f"Epoch {current_epoch + 1}: Stepped ReduceLROnPlateau. Current LR: {current_lr:.2e}")
            else:
                # Fallback for other schedulers if configured differently (unlikely with current setup)
                log.warning(f"Scheduler is not ReduceLROnPlateau ({type(scheduler)}). Attempting standard step().")
                try:
                    scheduler.step() # Assumes schedulers without metric argument in step
                    current_lr = optimizer.param_groups[0]['lr']
                    self.log('lr', current_lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True) # Log LR
                except TypeError:
                     log.error(f"Scheduler {type(scheduler)} requires a metric for step(), but none provided in standard step.")


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Generates embeddings for prediction."""
        # Handle different batch types (e.g., tuple vs tensor)
        if isinstance(batch, (list, tuple)) and len(batch) > 0 and isinstance(batch[0], torch.Tensor):
            embeddings = batch[0]
        elif isinstance(batch, torch.Tensor):
            embeddings = batch
        else:
            log.error(f"Unsupported batch type in predict_step: {type(batch)}")
            raise ValueError(f"Unsupported batch type: {type(batch)}")

        # Ensure input is on the correct device
        embeddings = embeddings.to(self.device)

        with torch.inference_mode():
            projected_embeddings = self(embeddings) # Apply model
            return projected_embeddings.cpu() # Return results on CPU