import torch
import torch.nn as nn
import lightning as L
from typing import List, Dict, Any, Tuple
from torchmetrics import Accuracy, F1Score, MatthewsCorrCoef, MetricCollection, MeanMetric, MaxMetric
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import OneCycleLR

# Define helper for classification metrics (CPU-based)
def compute_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    stage: str
) -> Dict[str, float]:
    """Computes classification metrics entirely on CPU."""
    metrics: Dict[str, float] = {}
    try:
        metrics[f"{stage}/acc"] = accuracy_score(targets, preds)
        metrics[f"{stage}/balanced_acc"] = balanced_accuracy_score(targets, preds)
        metrics[f"{stage}/precision"] = precision_score(targets, preds, average="macro", zero_division=0)
        metrics[f"{stage}/recall"] = recall_score(targets, preds, average="macro", zero_division=0)
        metrics[f"{stage}/f1"] = f1_score(targets, preds, average="macro", zero_division=0)
        metrics[f"{stage}/mcc"] = matthews_corrcoef(targets, preds)
    except Exception as e:
        print(f"{stage} metric calculation failed: {e}")
        for name in ("acc", "balanced_acc", "precision", "recall", "f1", "mcc"):
            metrics[f"{stage}/{name}"] = 0.0
    return metrics

class CATHeClassifier(L.LightningModule):
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
        
        # Buffers for storing predictions and targets for epoch-end metrics
        self._val_outputs: List[Dict[str, torch.Tensor]] = []
        self._test_outputs: List[Dict[str, torch.Tensor]] = []
        
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
        
        # Move tensors to CPU to save memory and store as tensors
        preds_cpu = preds.detach().cpu()
        labels_cpu = y.detach().cpu()
        loss_cpu = loss.detach().cpu()

        # Update loss metric
        self.val_loss(loss_cpu)

        # Store for end-of-epoch metric calculation
        self._val_outputs.append({"preds": preds_cpu, "labels": labels_cpu})

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits, loss, preds = self.model_step(batch)
        
        # Move tensors to CPU to save memory and store as tensors
        preds_cpu = preds.detach().cpu()
        labels_cpu = y.detach().cpu()

        # Store for end-of-epoch calculation
        self._test_outputs.append({"preds": preds_cpu, "labels": labels_cpu})

    def on_train_epoch_end(self) -> None:
        # Only log training loss for efficiency
        train_loss = self.train_loss.compute()
        self.log("train/loss_epoch", train_loss, prog_bar=True)
        self.train_loss.reset()

    def on_validation_epoch_end(self) -> None:
        val_loss = self.val_loss.compute()

        # Aggregate predictions and labels
        preds_cpu = torch.cat([o["preds"] for o in self._val_outputs]).numpy()
        labels_cpu = torch.cat([o["labels"] for o in self._val_outputs]).numpy()

        # Compute metrics
        metrics = compute_classification_metrics(preds_cpu, labels_cpu, stage="val")
        metrics["val/loss"] = val_loss

        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset for next epoch
        self.val_loss.reset()
        self._val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        # Aggregate predictions and labels
        preds_cpu = torch.cat([o["preds"] for o in self._test_outputs]).numpy()
        labels_cpu = torch.cat([o["labels"] for o in self._test_outputs]).numpy()

        # Compute metrics
        metrics = compute_classification_metrics(preds_cpu, labels_cpu, stage="test")

        # Log metrics
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Reset for next epoch
        self._test_outputs.clear()

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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Inference step for predictions."""
        if isinstance(batch, (list, tuple)):
            x = batch[0]
        else:
            x = batch
        x = x.to(self.device)
        with torch.no_grad():
            logits = self(x)
        return logits.cpu()