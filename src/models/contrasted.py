import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging

# Use standard logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Optional: For potentially faster distance calculations if available
try:
    import torch_cluster # pyg
    def pairwise_distance_optimized(x, y):
        # Using squared Euclidean distance
        return torch.cdist(x, y, p=2.0).pow(2)
except ImportError:
    log.info("torch_cluster not found. Using torch.cdist for pairwise distances.")
    def pairwise_distance_optimized(x, y):
        # Fallback using native PyTorch - ensure squared Euclidean
        return torch.cdist(x, y, p=2.0).pow(2)


class BatchHardMiner:
    """
    Implements Batch Hard Mining strategy for triplet selection within a batch.

    For each anchor, selects the hardest positive (farthest) and hardest negative
    (closest) sample within the batch.
    """
    def __init__(self, distance_metric_func=pairwise_distance_optimized):
        self.distance_metric = distance_metric_func

    def __call__(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Selects batch-hard triplets.

        Args:
            embeddings: Tensor of embeddings (batch_size, embedding_dim).
            labels: Tensor of integer labels (batch_size,).

        Returns:
            A tuple containing the indices of (anchors, positives, negatives)
            that form valid hard triplets within the batch. Returns empty tensors
            if no valid triplets can be formed.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)

        # Calculate pairwise distances (squared Euclidean)
        dist_mat = self.distance_metric(embeddings, embeddings)

        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_not_equal = ~labels_equal
        # Mask out diagonal (distance to self)
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        labels_equal.masked_fill_(identity_mask, False) # Cannot be positive of oneself

        # Find hardest positive for each anchor
        # Set non-positive distances to negative infinity
        hardest_pos_dist, hardest_pos_idx = torch.max(
            dist_mat * labels_equal.float(), # Zero out non-positives
            dim=1
        )

        # Find hardest negative for each anchor
        # Set non-negative distances to positive infinity
        neg_dists = dist_mat.clone()
        neg_dists[labels_equal] = float('inf') # Ignore positives
        neg_dists[identity_mask] = float('inf') # Ignore self
        hardest_neg_dist, hardest_neg_idx = torch.min(neg_dists, dim=1)

        # Create masks for valid triplets
        # Check 1: Ensure hardest positive was found (at least one other sample with same label exists)
        # Check 2: Ensure hardest negative was found (at least one sample with different label exists)
        # Check 3: Ensure the triplet contributes to loss (hard_pos_dist > hard_neg_dist, before margin)
        # Note: TripletLoss handles the margin, so we only need hard_pos > hard_neg conceptually,
        # but the loss function itself will filter trivial triplets (loss <= 0).

        # Find anchors where both a valid positive and negative were found
        valid_pos_mask = hardest_pos_dist > 0 # Needs a positive pair distance > 0
        valid_neg_mask = hardest_neg_dist != float('inf') # Needs a finite negative distance
        valid_anchor_mask = valid_pos_mask & valid_neg_mask

        # Get the indices for the valid hard triplets
        anchor_indices = torch.where(valid_anchor_mask)[0]
        positive_indices = hardest_pos_idx[anchor_indices]
        negative_indices = hardest_neg_idx[anchor_indices]

        if len(anchor_indices) == 0:
             log.debug("No valid triplets found in this batch.")
             return torch.empty(0, dtype=torch.long, device=device), \
                    torch.empty(0, dtype=torch.long, device=device), \
                    torch.empty(0, dtype=torch.long, device=device)

        return anchor_indices, positive_indices, negative_indices


class ContrastiveCATHeModel(pl.LightningModule):
    """
    PyTorch Lightning module for contrastive learning using a projection head
    and Batch Hard Triplet Loss.
    """
    # Constants for evaluation
    DEFAULT_KNN_NEIGHBORS = 5
    DEFAULT_VAL_SPLIT_RATIO = 0.7
    DEFAULT_TEST_CV_FOLDS = 5

    def __init__(
        self,
        input_embedding_dim: int,
        projection_hidden_dims: List[int] = [512],
        output_embedding_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        triplet_margin: float = 1.0,
        use_layer_norm: bool = True,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        # Evaluation parameters
        knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
        knn_val_split_ratio: float = DEFAULT_VAL_SPLIT_RATIO,
        knn_test_cv_folds: int = DEFAULT_TEST_CV_FOLDS,
    ):
        """
        Initializes the contrastive learning model.

        Args:
            input_embedding_dim: Dimension of the input pre-computed embeddings.
            projection_hidden_dims: List of hidden layer sizes for the MLP projection network.
            output_embedding_dim: Dimension of the final output embedding space.
            dropout: Dropout probability in the projection network.
            learning_rate: Optimizer learning rate.
            weight_decay: Optimizer weight decay (L2 regularization).
            triplet_margin: Margin for the triplet loss function.
            use_layer_norm: Whether to use Layer Normalization in the projection head.
            lr_scheduler_config: Configuration dictionary for the LR scheduler (e.g., ReduceLROnPlateau).
                                 Example: {"monitor": "val/loss", "factor": 0.5, "patience": 5}
            knn_neighbors: Number of neighbors for kNN evaluation.
            knn_val_split_ratio: Ratio to split validation data for kNN training (e.g., 0.7 means 70% train, 30% test).
            knn_test_cv_folds: Number of folds for cross-validation during kNN testing.
        """
        super().__init__()
        # Store hyperparameters (automatically logs them)
        self.save_hyperparameters()

        # Build the projection network (MLP)
        self.projection = self._build_projection_network(
            input_dim=self.hparams.input_embedding_dim,
            hidden_dims=self.hparams.projection_hidden_dims,
            output_dim=self.hparams.output_embedding_dim,
            dropout=self.hparams.dropout,
            use_layer_norm=self.hparams.use_layer_norm
        )
        self._init_weights()

        # Loss function: Standard TripletMarginLoss using squared Euclidean distance (p=2)
        self.loss_fn = nn.TripletMarginLoss(
            margin=self.hparams.triplet_margin,
            p=2.0,  # Euclidean distance
            reduction='mean' # Average loss over the batch of valid triplets
        )

        # Triplet miner
        self.miner = BatchHardMiner()

        # Lists to store validation/test outputs for epoch-end evaluation
        self._val_outputs = []
        self._test_outputs = []

    def _build_projection_network(self, input_dim: int, hidden_dims: List[int],
                                  output_dim: int, dropout: float, use_layer_norm: bool) -> nn.Sequential:
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

        # Final projection layer (no activation/norm here, done in forward)
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming Normal for ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input embeddings through the projection network and applies L2 normalization.

        Args:
            x: Input protein embeddings tensor (batch_size, input_embedding_dim).

        Returns:
            Projected and L2-normalized embeddings tensor (batch_size, output_embedding_dim).
        """
        projected_embeddings = self.projection(x)
        # L2 normalization is crucial for many metric learning losses
        normalized_embeddings = F.normalize(projected_embeddings, p=2, dim=1)
        return normalized_embeddings

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Performs a single training step: projects embeddings, mines triplets, calculates loss.
        """
        embeddings, labels = batch

        # Project embeddings into the learned space
        projected_embeddings = self(embeddings)

        # Mine hard triplets within the batch
        anchor_idx, positive_idx, negative_idx = self.miner(projected_embeddings, labels)

        # If no valid triplets found in the batch, return zero loss
        if len(anchor_idx) == 0:
            # Consolidated logging
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            self.log_dict({
                "train/loss": zero_loss,
                "train/active_triplets": 0.0
            }, on_step=True, on_epoch=True, prog_bar=True)
            return zero_loss

        # Select the embeddings corresponding to the mined triplet indices
        anchor_emb = projected_embeddings[anchor_idx]
        positive_emb = projected_embeddings[positive_idx]
        negative_emb = projected_embeddings[negative_idx]

        # Calculate triplet loss
        loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)

        # Consolidated logging
        self.log_dict({
            "train/loss": loss,
            "train/active_triplets": float(len(anchor_idx))
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def _log_zero_metrics(self, prefix: str, metrics: List[str], prog_bar: bool = False) -> None:
        """Helper to log zero values for metrics when evaluation is skipped."""
        metric_dict = {f"{prefix}/{metric}": 0.0 for metric in metrics}
        self.log_dict(metric_dict, prog_bar=prog_bar)

    def _collect_and_clear_outputs(self, outputs: List[Dict[str, torch.Tensor]]) -> Tuple[np.ndarray, np.ndarray]:
        """Collects embeddings and labels from outputs list and clears it."""
        all_embeddings = torch.cat([x["embeddings"] for x in outputs]).numpy()
        all_labels = torch.cat([x["labels"] for x in outputs]).numpy()
        outputs.clear()  # Free memory
        return all_embeddings, all_labels

    # --- Validation ---
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Projects validation embeddings and stores them with labels for epoch-end evaluation."""
        embeddings, labels = batch
        projected_embeddings = self(embeddings)
        # Store outputs: move to CPU to avoid GPU memory buildup across steps
        self._val_outputs.append({
            "embeddings": projected_embeddings.detach().cpu(),
            "labels": labels.detach().cpu()
        })

    def on_validation_epoch_end(self) -> None:
        """Computes kNN classification metrics on the validation set at the end of the epoch."""
        if not self._val_outputs:
            log.warning("Validation epoch end called but no outputs were collected.")
            return

        # Collate embeddings and labels from all validation steps
        all_embeddings, all_labels = self._collect_and_clear_outputs(self._val_outputs)

        # Ensure enough data for split and kNN
        min_samples_for_split = max(2, self.hparams.knn_neighbors) # Need at least k neighbors + 1 for train/test
        if len(all_embeddings) < min_samples_for_split * 2 or len(np.unique(all_labels)) < 2:
            log.warning(f"Skipping validation kNN: Insufficient samples ({len(all_embeddings)}) "
                        f"or classes ({len(np.unique(all_labels))}).")
            self._log_zero_metrics("val", ["knn_acc", "knn_balanced_acc"], prog_bar=True)
            return

        try:
            # Split validation data itself into train/test for kNN
            knn_train_embeds, knn_test_embeds, knn_train_labels, knn_test_labels = train_test_split(
                all_embeddings,
                all_labels,
                train_size=self.hparams.knn_val_split_ratio,
                stratify=all_labels,
                random_state=42
            )

            # Ensure the test split is usable
            if len(knn_test_embeds) == 0 or len(np.unique(knn_train_labels)) < 2:
                 log.warning(f"Skipping validation kNN: Split resulted in unusable train/test sets.")
                 self._log_zero_metrics("val", ["knn_acc", "knn_balanced_acc"], prog_bar=True)
                 return

            # Compute kNN metrics using the helper
            metrics = self._compute_knn_metrics(
                knn_train_embeds, knn_train_labels,
                knn_test_embeds, knn_test_labels,
                self.hparams.knn_neighbors
            )

            # Log validation kNN metrics
            self.log_dict({
                "val/knn_acc": metrics["accuracy"],
                "val/knn_balanced_acc": metrics["balanced_accuracy"]
            }, prog_bar=True)
            
            log.info(f"Validation kNN - Acc: {metrics['accuracy']:.4f}, Balanced Acc: {metrics['balanced_accuracy']:.4f}")

        except Exception as e:
            log.error(f"Error during validation kNN evaluation: {e}", exc_info=True)
            self._log_zero_metrics("val", ["knn_acc", "knn_balanced_acc"], prog_bar=True)

    # --- Testing ---
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Projects test embeddings and stores them with labels for epoch-end evaluation."""
        embeddings, labels = batch
        projected_embeddings = self(embeddings)
        self._test_outputs.append({
            "embeddings": projected_embeddings.detach().cpu(),
            "labels": labels.detach().cpu()
        })

    def on_test_epoch_end(self) -> None:
        """Computes kNN classification metrics using k-fold CV on the test set."""
        if not self._test_outputs:
            log.warning("Test epoch end called but no outputs were collected.")
            return

        # Collate embeddings and labels
        all_embeddings, all_labels = self._collect_and_clear_outputs(self._test_outputs)

        n_samples = len(all_embeddings)
        n_classes = len(np.unique(all_labels))
        n_folds = self.hparams.knn_test_cv_folds

        # Ensure enough data for k-fold CV and kNN
        min_samples_for_cv = max(n_folds, self.hparams.knn_neighbors + 1)
        if n_samples < min_samples_for_cv or n_classes < 2:
            log.warning(f"Skipping test kNN CV: Insufficient samples ({n_samples}) "
                        f"or classes ({n_classes}) for {n_folds}-fold CV.")
            self._log_zero_metrics("test", ["knn_acc_cv_mean", "knn_balanced_acc_cv_mean", "knn_f1_macro_cv_mean"])
            return

        try:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_metrics = {"accuracy": [], "balanced_accuracy": [], "f1_macro": []}

            log.info(f"Starting {n_folds}-fold kNN Cross-Validation on test set...")
            for fold_idx, (train_index, test_index) in enumerate(skf.split(all_embeddings, all_labels)):
                knn_train_embeds, knn_test_embeds = all_embeddings[train_index], all_embeddings[test_index]
                knn_train_labels, knn_test_labels = all_labels[train_index], all_labels[test_index]

                # Ensure fold is usable
                if len(knn_test_embeds) == 0 or len(np.unique(knn_train_labels)) < 2:
                    log.warning(f"Skipping Fold {fold_idx+1}/{n_folds} in test kNN CV: Insufficient data.")
                    continue

                # Compute metrics for the fold
                metrics = self._compute_knn_metrics(
                    knn_train_embeds, knn_train_labels,
                    knn_test_embeds, knn_test_labels,
                    self.hparams.knn_neighbors
                )
                
                # Collect metrics for this fold
                for metric_name, value in metrics.items():
                    if metric_name in fold_metrics:
                        fold_metrics[metric_name].append(value)
                
                log.debug(f"  Fold {fold_idx+1}/{n_folds} - Acc: {metrics['accuracy']:.4f}, BalAcc: {metrics['balanced_accuracy']:.4f}")

            # Calculate mean and std dev over folds if metrics were collected
            if not fold_metrics["accuracy"]:  # Check if any folds were successful
                log.warning("No successful folds completed during test kNN CV.")
                self._log_zero_metrics("test", ["knn_acc_cv_mean", "knn_balanced_acc_cv_mean", "knn_f1_macro_cv_mean"])
                return
            
            # Calculate and log statistics from folds
            result_metrics = {}
            for metric_name, values in fold_metrics.items():
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
                result_metrics[f"test/knn_{metric_name}_cv_mean"] = mean_val
                result_metrics[f"test/knn_{metric_name}_cv_std"] = std_val
            
            self.log_dict(result_metrics)

            # Display summary
            log.info(f"Test kNN CV Results ({n_folds}-fold):")
            log.info(f"  Accuracy:          {result_metrics['test/knn_accuracy_cv_mean']:.4f} +/- {result_metrics['test/knn_accuracy_cv_std']:.4f}")
            log.info(f"  Balanced Accuracy: {result_metrics['test/knn_balanced_accuracy_cv_mean']:.4f} +/- {result_metrics['test/knn_balanced_accuracy_cv_std']:.4f}")
            log.info(f"  Macro F1-Score:    {result_metrics['test/knn_f1_macro_cv_mean']:.4f} +/- {result_metrics['test/knn_f1_macro_cv_std']:.4f}")

        except Exception as e:
            log.error(f"Error during test kNN CV evaluation: {e}", exc_info=True)
            self._log_zero_metrics("test", ["knn_acc_cv_mean", "knn_balanced_acc_cv_mean", "knn_f1_macro_cv_mean"])

    def _compute_knn_metrics(self, train_embeds: np.ndarray, train_labels: np.ndarray,
                             test_embeds: np.ndarray, test_labels: np.ndarray,
                             n_neighbors: int) -> Dict[str, float]:
        """Helper function to fit kNN and compute classification metrics."""
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(train_embeds, train_labels)
        y_pred = knn.predict(test_embeds)

        return {
            "accuracy": float(accuracy_score(test_labels, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, y_pred, adjusted=False)),
            "f1_macro": float(f1_score(test_labels, y_pred, average='macro', zero_division=0))
        }

    # --- Optimizer and Scheduler ---
    def configure_optimizers(self):
        """Configures the AdamW optimizer and an optional LR scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        # Configure optional learning rate scheduler
        if not self.hparams.lr_scheduler_config:
            log.info("No LR scheduler configured.")
            return optimizer
            
        scheduler_params = self.hparams.lr_scheduler_config
        monitor = scheduler_params.get("monitor", "val/knn_balanced_acc")
        mode = scheduler_params.get("mode", "max")

        # Example: Using ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=scheduler_params.get("factor", 0.5),
            patience=scheduler_params.get("patience", 5),
            min_lr=scheduler_params.get("min_lr", 1e-7),
            verbose=True
        )
        
        log.info(f"Using ReduceLROnPlateau LR scheduler, monitoring '{monitor}' (mode: {mode}).")
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": False,
            }
        }

    # --- Prediction ---
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Projects input embeddings for inference."""
        if isinstance(batch, (tuple, list)):
            # Assume first element is the embedding if batch is tuple/list
            embeddings = batch[0]
        elif isinstance(batch, torch.Tensor):
            embeddings = batch
        else:
            raise TypeError(f"Unsupported batch type for prediction: {type(batch)}")

        return self(embeddings.to(self.device))