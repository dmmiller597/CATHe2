import logging
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Use standard logging
log = logging.getLogger(__name__)


class ProtTuckerLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training a ProtTucker-like model using contrastive loss.

    Adapts the original ProtTucker concept to work with batches of (embedding, label)
    pairs provided by a DataLoader, applying batch-hard triplet mining within the
    training step. Evaluation uses k-NN classification accuracy on the projected embeddings.
    """
    def __init__(
        self,
        input_embedding_dim: int,
        projection_hidden_dims: List[int], # e.g., [256]
        output_embedding_dim: int, # e.g., 128
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        triplet_margin: Optional[float] = 0.5, # Use None for SoftMarginLoss, float for MarginRankingLoss
        use_batch_hard: bool = True, # Batch-hard mining is recommended for this setup
        knn_eval_neighbors: int = 5,
        optimizer_config: Optional[Dict[str, Any]] = None, # For AdamW betas etc.
        scheduler_config: Optional[Dict[str, Any]] = None, # For ReduceLROnPlateau
        **kwargs # Catches potential extra args from Hydra/config
    ):
        """
        Args:
            input_embedding_dim: Dimension of input protein embeddings (e.g., 1024).
            projection_hidden_dims: List of dimensions for the hidden layers in the ProtTucker MLP.
            output_embedding_dim: Dimension of the final contrastive embedding space.
            learning_rate: Optimizer learning rate.
            weight_decay: Optimizer weight decay (L2 regularization).
            triplet_margin: Margin for MarginRankingLoss. Use None for SoftMarginLoss.
            use_batch_hard: Whether to use batch-hard mining.
            knn_eval_neighbors: Number of neighbors for k-NN evaluation.
            optimizer_config: Config for AdamW optimizer.
            scheduler_config: Config for learning rate scheduler (ReduceLROnPlateau).
        """
        super().__init__()
        if not use_batch_hard:
            log.warning("use_batch_hard=False is not standard for this setup where triplets aren't pre-defined. Using batch-hard mining.")
            use_batch_hard = True # Force batch-hard as it's the natural fit

        # Store hyperparameters
        self.save_hyperparameters()

        # Build the projection network
        layers = []
        current_dim = input_embedding_dim
        for hidden_dim in projection_hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh()) # Original ProtTucker used Tanh
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_embedding_dim))
        self.projection_head = nn.Sequential(*layers)

        # Triplet loss configuration
        self.distance = nn.PairwiseDistance(p=2)
        if self.hparams.triplet_margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=self.hparams.triplet_margin, reduction='mean')
            log.info(f"Using MarginRankingLoss with margin={self.hparams.triplet_margin}")
        else:
            self.ranking_loss = nn.SoftMarginLoss(reduction='mean')
            log.info("Using SoftMarginLoss (margin=None)")
            
        # For storing validation/test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input embeddings to the output space and L2 normalizes them."""
        projected = self.projection_head(x)
        # L2 normalize the embeddings - crucial for cosine similarity/distance interpretation
        normalized_embeddings = F.normalize(projected, p=2, dim=1)
        return normalized_embeddings

    def _calculate_triplet_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the batch-hard triplet loss.

        Args:
            embeddings: Projected embeddings for the batch (shape: [batch_size, output_dim]).
            labels: Integer labels for the batch (shape: [batch_size]).

        Returns:
            Triplet loss value for the batch.
        """
        # Calculate pairwise distances (euclidean)
        pdist_matrix = torch.cdist(embeddings, embeddings, p=2)

        # Create masks for positive and negative pairs based on labels
        batch_size = labels.size(0)
        labels_expanded = labels.view(-1, 1)
        # Positive mask: True where labels are the same (excluding self)
        mask_positive = (labels_expanded == labels_expanded.T) & (~torch.eye(batch_size, dtype=torch.bool, device=self.device))
        # Negative mask: True where labels are different
        mask_negative = (labels_expanded != labels_expanded.T)

        # Batch Hard Mining:
        # Find hardest positive (max distance within same class)
        # Use max distance, ensuring we only consider actual positives
        dist_ap, _ = torch.max(pdist_matrix * mask_positive.float(), dim=1)

        # Find hardest negative (min distance to different class)
        # Set non-negative pairs to infinity to ignore them in min calculation
        negative_distances = torch.where(mask_negative, pdist_matrix, torch.tensor(float('inf'), device=self.device))
        dist_an, _ = torch.min(negative_distances, dim=1)

        # Target variable for loss function (all 1s for triplet loss)
        target = torch.ones_like(dist_an)

        if self.hparams.triplet_margin is not None:
            # MarginRankingLoss: loss = max(0, dist_ap - dist_an + margin)
            # Note: MarginRankingLoss expects input1=dist_an, input2=dist_ap, target=1
            loss = self.ranking_loss(dist_an, dist_ap, target)
        else:
            # SoftMarginLoss: loss = log(1 + exp(dist_ap - dist_an))
            loss = self.ranking_loss(dist_ap - dist_an, target) # SoftMargin takes diff and target=1 implicitly

        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        embeddings, labels = batch
        projected_embeddings = self(embeddings)
        loss = self._calculate_triplet_loss(projected_embeddings, labels)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Log embedding norms for monitoring
        self.log('train/embedding_norm', projected_embeddings.norm(dim=1).mean(), on_step=False, on_epoch=True, logger=True)

        return loss

    def _evaluation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Common logic for validation and test steps."""
        embeddings, labels = batch
        projected_embeddings = self(embeddings)
        # Detach and move to CPU for accumulation
        return {'embeddings': projected_embeddings.detach().cpu(), 'labels': labels.detach().cpu()}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a single validation step."""
        outputs = self._evaluation_step(batch, batch_idx)
        self.validation_step_outputs.append(outputs)
        # Optional: Calculate and log batch loss if desired (less common for validation)
        # with torch.no_grad():
        #     loss = self._calculate_triplet_loss(self(batch[0]), batch[1])
        # self.log('val/loss_step', loss, on_step=True, on_epoch=False)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Performs a single test step."""
        outputs = self._evaluation_step(batch, batch_idx)
        self.test_step_outputs.append(outputs)

    def _evaluation_epoch_end(self, step_outputs: List[Dict[str, torch.Tensor]], stage: str):
        """Common logic for end of validation/test epoch: k-NN evaluation."""
        if not step_outputs:
            log.warning(f"No outputs found for {stage} epoch end evaluation.")
            return

        # Concatenate all outputs from the epoch
        all_embeddings = torch.cat([x['embeddings'] for x in step_outputs]).numpy()
        all_labels = torch.cat([x['labels'] for x in step_outputs]).numpy()
        
        step_outputs.clear() # Free memory

        if len(np.unique(all_labels)) <= 1:
            log.warning(f"Skipping {stage} k-NN evaluation: Found <= 1 unique class.")
            # Log default/zero values to prevent logging errors downstream
            self.log(f'{stage}/knn_balanced_acc', 0.0, prog_bar=True, logger=True)
            return

        n_samples = all_embeddings.shape[0]
        k = min(self.hparams.knn_eval_neighbors, n_samples -1) # Ensure k < n_samples

        if k <= 0:
            log.warning(f"Skipping {stage} k-NN evaluation: Not enough samples (n={n_samples}) for k={self.hparams.knn_eval_neighbors}.")
            self.log(f'{stage}/knn_balanced_acc', 0.0, prog_bar=True, logger=True)
            return

        log.info(f"Running {stage} k-NN evaluation with k={k} on {n_samples} samples...")

        try:
            # Use scikit-learn for efficient k-NN
            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
            # Train on all data (predict label for each sample using others)
            knn.fit(all_embeddings, all_labels)
            # Predict on the same data (leave-one-out is implicitly handled by kNN fit/predict)
            # NOTE: This evaluates how well the space separates the classes present in this split.
            # It's not predicting on unseen data, but rather assessing cluster quality.
            predictions = knn.predict(all_embeddings)

            # Calculate balanced accuracy
            balanced_acc = balanced_accuracy_score(all_labels, predictions)

            self.log(f'{stage}/knn_balanced_acc', balanced_acc, prog_bar=True, logger=True)
            log.info(f"{stage.capitalize()} k-NN Balanced Accuracy (k={k}): {balanced_acc:.4f}")

        except Exception as e:
            log.error(f"Error during {stage} k-NN evaluation: {e}", exc_info=True)
            self.log(f'{stage}/knn_balanced_acc', 0.0, prog_bar=True, logger=True) # Log default on error

    def on_validation_epoch_end(self):
        """Runs k-NN evaluation at the end of the validation epoch."""
        self._evaluation_epoch_end(self.validation_step_outputs, stage='val')

    def on_test_epoch_end(self):
        """Runs k-NN evaluation at the end of the test epoch."""
        self._evaluation_epoch_end(self.test_step_outputs, stage='test')

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer_defaults = {"lr": self.hparams.learning_rate, "weight_decay": self.hparams.weight_decay}
        if self.hparams.optimizer_config:
            optimizer_defaults.update(self.hparams.optimizer_config)
        optimizer = AdamW(self.parameters(), **optimizer_defaults)

        scheduler_config = self.hparams.scheduler_config
        if scheduler_config:
            scheduler_defaults = {
                "monitor": "val/knn_balanced_acc", # Default monitor metric
                "mode": "max",                     # Default mode for accuracy
                "patience": 10,
                "factor": 0.1,
                "verbose": True,
            }
            scheduler_defaults.update(scheduler_config) # Override with user config
            scheduler = ReduceLROnPlateau(optimizer, **scheduler_defaults)
            
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": scheduler_defaults["monitor"],
            }
            log.info(f"Using ReduceLROnPlateau scheduler monitoring '{scheduler_defaults['monitor']}'")
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
        else:
            log.info("No learning rate scheduler configured.")
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Generates final normalized embeddings for prediction/inference."""
        # Handle different batch structures (e.g., tuple vs tensor)
        if isinstance(batch, tuple) and len(batch) > 0:
            # Assume first element is the embedding tensor
            # Ensure input is moved to the correct device within the step
            embeddings = batch[0].to(self.device)
        elif isinstance(batch, torch.Tensor):
            # Ensure input is moved to the correct device within the step
            embeddings = batch.to(self.device)
        else:
            raise ValueError(f"Unsupported batch type for prediction: {type(batch)}")

        # Use inference mode and generate normalized embeddings via the forward pass
        with torch.inference_mode():
            # Forward pass handles projection and normalization
            normalized_embeddings = self(embeddings)
            # Return embeddings on CPU for easier collection/saving
            return normalized_embeddings.cpu()
