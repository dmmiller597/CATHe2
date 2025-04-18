import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Dict, Any, Tuple
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, MetricCollection, MeanMetric, MaxMetric
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from torch.optim.lr_scheduler import OneCycleLR

class CATHeClassifier(pl.LightningModule):
    """PyTorch Lightning module for CATH superfamily classification."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ):
        """
        Initialize the CATH classifier.

        Args:
            embedding_dim: Dimension of input protein embeddings
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of CATH superfamily classes
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization) for optimizer
        """
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        # Build layers
        layers = []
        in_features = embedding_dim
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size, bias=True),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout)
            ])
            in_features = hidden_size

        layers.append(nn.Linear(in_features, num_classes))
        self.model = nn.Sequential(*layers)
        self._init_weights()

        # Define metrics - avoid using full num_classes for torchmetrics initialization
        # This prevents large confusion matrices from being created in GPU memory
        self.num_classes = num_classes
        
        # Use MeanMetric for tracking loss values
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric().to('cpu')
        
        # Create prediction and target buffers for custom metric calculation
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        
        # Loss criterion
        self.criterion = nn.CrossEntropyLoss()

    def _init_weights(self) -> None:
        """Initialize network weights using Kaiming initialization with GELU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_in',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return logits, loss, preds

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Track loss during training - move to CPU first
        self.train_loss(loss.detach().cpu())
        
        # Log with less frequency
        if batch_idx % 50 == 0:  # Adjust logging frequency as needed
            self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Move tensors to CPU and store as numpy arrays to save memory
        preds_cpu = preds.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        loss_cpu = loss.detach().cpu()
        
        # Update loss metric
        self.val_loss(loss_cpu)
        
        # Store predictions and targets for end-of-epoch metric calculation
        self.val_preds.append(preds_cpu)
        self.val_targets.append(y_cpu)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Move tensors to CPU and store as numpy arrays
        preds_cpu = preds.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        
        # Store for end-of-epoch calculation
        self.test_preds.append(preds_cpu)
        self.test_targets.append(y_cpu)

    def on_train_epoch_end(self) -> None:
        # Only log training loss for efficiency
        train_loss = self.train_loss.compute()
        self.log("train/loss_epoch", train_loss, prog_bar=True)
        self.train_loss.reset()

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()
        
        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.val_preds)
        all_targets = np.concatenate(self.val_targets)
        
        # Calculate accuracy manually (memory efficient)
        accuracy = np.mean(all_preds == all_targets)
        
        # Calculate balanced accuracy manually
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        
        # Log metrics
        self.log_dict(
            {
                "val/loss": val_loss,
                "val/acc": accuracy,
                "val/balanced_acc": balanced_acc
            },
            prog_bar=True,
            sync_dist=True
        )
        
        # Reset storage
        self.val_loss.reset()
        self.val_preds = []
        self.val_targets = []

    def on_test_epoch_end(self) -> None:
        # Concatenate all predictions and targets
        all_preds = np.concatenate(self.test_preds)
        all_targets = np.concatenate(self.test_targets)
        
        # Calculate metrics manually (memory efficient)
        metrics = {
            "acc": np.mean(all_preds == all_targets),
            "balanced_acc": balanced_accuracy_score(all_targets, all_preds)
        }
        
        # Only calculate F1 if requested (expensive for 3000+ classes)
        try:
            # Use macro averaging to handle class imbalance
            metrics["f1"] = f1_score(all_targets, all_preds, average='macro')
        except Exception as e:
            self.print(f"F1 calculation failed: {str(e)}")
        
        # Try to calculate MCC (can be memory intensive)
        try:
            # Remove the 'average' parameter - matthews_corrcoef doesn't accept it
            metrics["mcc"] = matthews_corrcoef(all_targets, all_preds)
        except Exception as e:
            # Use self.print instead of self.log for error messages
            self.print(f"MCC calculation failed: {str(e)}")
        
        # Log all metrics
        for name, value in metrics.items():
            self.log(f"test/{name}", value, sync_dist=True)
        
        # Reset storage
        self.test_preds = []
        self.test_targets = []

    def configure_optimizers(self):
        """Configure optimizer with learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        # Use OneCycleLR scheduler over the estimated total training steps
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }