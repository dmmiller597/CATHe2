import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor # Explicit type hinting

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
        projection_hidden_dims: List[int] = [512],
        output_embedding_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        triplet_margin: float = 0.5,
        use_layer_norm: bool = True,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        knn_val_neighbors: int = 1,
        knn_test_neighbors: int = 5,
        knn_test_cv_folds: int = 5,
        val_max_samples: int = 10000,
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
            knn_test_neighbors: Number of neighbors for test kNN.
            knn_test_cv_folds: Number of CV folds for testing.
            val_max_samples: Maximum validation samples to use for kNN.
        """
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters()

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
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/active_triplets', float(active_triplets), on_step=False, on_epoch=True, sync_dist=True)
        
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
        """Computes k-NN validation metrics."""
        if not self._val_outputs:
            self.log("val/knn_acc", 0.0, prog_bar=True, sync_dist=True)
            self.log("val/knn_balanced_acc", 0.0, prog_bar=True, sync_dist=True)
            return

        try:
            # Concatenate embeddings and labels
            all_embeddings = torch.cat([x["embeddings"] for x in self._val_outputs])
            all_labels = torch.cat([x["labels"] for x in self._val_outputs])
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

    # --- Testing ---
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        """Stores projected test embeddings for end-of-epoch evaluation."""
        embeddings, labels = batch
        with torch.inference_mode():
            projected_embeddings = self(embeddings)
        
        # Move to CPU to free GPU memory
        self._test_outputs.append({
            "embeddings": projected_embeddings.detach().cpu(),
            "labels": labels.detach().cpu()
        })

    def on_test_epoch_end(self) -> None:
        """Performs k-fold CV evaluation using k-NN classifier."""
        if not self._test_outputs:
            self.log_dict({
                "test/cv_acc": 0.0,
                "test/cv_balanced_acc": 0.0,
                "test/cv_f1_macro": 0.0
            }, sync_dist=True)
            return

        try:
            # Collect all embeddings and labels
            all_embeddings = torch.cat([x["embeddings"] for x in self._test_outputs]).numpy()
            all_labels = torch.cat([x["labels"] for x in self._test_outputs]).numpy()
            self._test_outputs.clear()  # Free memory

            # Perform k-fold cross-validation
            k = self.hparams.knn_test_neighbors
            n_folds = self.hparams.knn_test_cv_folds
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            accuracies = []
            balanced_accuracies = []
            f1_scores = []
            
            for train_idx, test_idx in skf.split(all_embeddings, all_labels):
                # Train k-NN classifier
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(all_embeddings[train_idx], all_labels[train_idx])
                
                # Predict and evaluate
                y_pred = knn.predict(all_embeddings[test_idx])
                y_true = all_labels[test_idx]
                
                accuracies.append(accuracy_score(y_true, y_pred))
                balanced_accuracies.append(balanced_accuracy_score(y_true, y_pred))
                f1_scores.append(f1_score(y_true, y_pred, average='macro', zero_division=0))
            
            # Log average metrics
            self.log_dict({
                "test/cv_acc": np.mean(accuracies),
                "test/cv_balanced_acc": np.mean(balanced_accuracies),
                "test/cv_f1_macro": np.mean(f1_scores)
            }, sync_dist=True)
            
        except Exception as e:
            log.error(f"Error in test metrics: {e}")
            self.log_dict({
                "test/cv_acc": 0.0,
                "test/cv_balanced_acc": 0.0,
                "test/cv_f1_macro": 0.0
            }, sync_dist=True)

    def configure_optimizers(self):
        """Configures optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        if not self.hparams.lr_scheduler_config:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.hparams.lr_scheduler_config.get("mode", "max"),
            factor=self.hparams.lr_scheduler_config.get("factor", 0.5),
            patience=self.hparams.lr_scheduler_config.get("patience", 5),
            min_lr=self.hparams.lr_scheduler_config.get("min_lr", 1e-7),
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.lr_scheduler_config.get("monitor", "val/knn_balanced_acc"),
                "interval": "epoch",
                "frequency": 1
            }
        }

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